//! T-minus event simulation core: agents predict musical events locally without a shared clock.
//!
//! # Overview
//!
//! In a distributed musical ensemble, agents do not share a master clock. Instead, each agent
//! predicts when the next beat, bar, or phrase will occur based on its local tempo model, then
//! listens to others and gradually converges through drift correction.
//!
//! ## Sections
//! - `event` — musical event types with prediction error tracking
//! - `phase` — continuous phase accumulator (beat / bar / phrase)
//! - `history` — ring-buffer of past events for tempo estimation
//! - `predictor` — local tempo model that predicts upcoming events
//! - `drift` — inter-agent drift detection and correction factors
//! - `clock` — top-level [`TMinusClock`] that ties everything together

// ─────────────────────────────────────────────────────────────────────────────
// event
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of musical event an agent predicts or observes.
#[derive(Debug, Clone, PartialEq)]
pub enum EventKind {
    /// A single beat pulse.
    Beat,
    /// A bar boundary (every `beats_per_bar` beats).
    Bar,
    /// A phrase boundary (every `bars_per_phrase` bars).
    Phrase,
    /// A user-defined event identified by an opaque byte tag.
    Custom(u8),
}

/// A single predicted (and optionally observed) musical event.
#[derive(Debug, Clone)]
pub struct Event {
    /// The kind of musical event.
    pub kind: EventKind,
    /// Wall time (seconds) when this prediction was made.
    pub predicted_at: f64,
    /// Wall time (seconds) when the event is expected to fire.
    pub scheduled_for: f64,
    /// Wall time (seconds) when the event actually fired, if it has.
    pub actual_at: Option<f64>,
}

impl Event {
    /// Create a new unfired event.
    pub fn new(kind: EventKind, predicted_at: f64, scheduled_for: f64) -> Self {
        Self {
            kind,
            predicted_at,
            scheduled_for,
            actual_at: None,
        }
    }

    /// Returns the timing error (actual − scheduled) in seconds, or `None` if not yet fired.
    pub fn error(&self) -> Option<f64> {
        self.actual_at.map(|a| a - self.scheduled_for)
    }

    /// Mark this event as having fired at `actual_time`.
    pub fn mark_fired(&mut self, actual_time: f64) {
        self.actual_at = Some(actual_time);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// phase
// ─────────────────────────────────────────────────────────────────────────────

/// Result of advancing the phase tracker by some number of beats.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseAdvanceResult {
    /// Number of beat boundaries crossed.
    pub beats_fired: u32,
    /// Number of bar boundaries crossed.
    pub bars_fired: u32,
    /// Number of phrase boundaries crossed.
    pub phrases_fired: u32,
}

/// Continuous phase accumulator that tracks position within a beat, bar, and phrase.
pub struct PhaseTracker {
    /// Global phrase phase in `[0.0, 1.0)`.
    pub phase: f64,
    /// Number of beats per bar (e.g. 4 for 4/4).
    pub beats_per_bar: u32,
    /// Number of bars per phrase (e.g. 8).
    pub bars_per_phrase: u32,
    /// Within-beat phase in `[0.0, 1.0)`.
    pub beat_phase: f64,
    /// Total accumulated beats (monotonically increasing).
    total_beats: f64,
}

impl PhaseTracker {
    /// Create a new `PhaseTracker` starting at phase zero.
    pub fn new(beats_per_bar: u32, bars_per_phrase: u32) -> Self {
        Self {
            phase: 0.0,
            beats_per_bar,
            bars_per_phrase,
            beat_phase: 0.0,
            total_beats: 0.0,
        }
    }

    /// Advance the tracker by `dt_beats` beats and return what boundaries were crossed.
    pub fn advance(&mut self, dt_beats: f64) -> PhaseAdvanceResult {
        let beats_per_phrase = self.beats_per_bar as f64 * self.bars_per_phrase as f64;

        let old_total_beats = self.total_beats;
        self.total_beats += dt_beats;

        let old_full_beats = old_total_beats.floor() as u64;
        let new_full_beats = self.total_beats.floor() as u64;
        let beats_fired = (new_full_beats - old_full_beats) as u32;

        let old_full_bars = (old_total_beats / self.beats_per_bar as f64).floor() as u64;
        let new_full_bars = (self.total_beats / self.beats_per_bar as f64).floor() as u64;
        let bars_fired = (new_full_bars - old_full_bars) as u32;

        let old_full_phrases = (old_total_beats / beats_per_phrase).floor() as u64;
        let new_full_phrases = (self.total_beats / beats_per_phrase).floor() as u64;
        let phrases_fired = (new_full_phrases - old_full_phrases) as u32;

        // Update beat_phase (fractional part of total_beats)
        self.beat_phase = self.total_beats - self.total_beats.floor();

        // Update global phrase phase
        self.phase = (self.total_beats / beats_per_phrase).fract();

        PhaseAdvanceResult {
            beats_fired,
            bars_fired,
            phrases_fired,
        }
    }

    /// Phase within the current bar, in `[0.0, 1.0)`.
    pub fn bar_phase(&self) -> f64 {
        let beat_in_bar = self.total_beats % self.beats_per_bar as f64;
        beat_in_bar / self.beats_per_bar as f64
    }

    /// Alias for `phase` — position within the current phrase `[0.0, 1.0)`.
    pub fn phrase_phase(&self) -> f64 {
        self.phase
    }

    /// Which beat within the current bar (0-indexed).
    pub fn beat_in_bar(&self) -> u32 {
        (self.total_beats.floor() as u64 % self.beats_per_bar as u64) as u32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// history
// ─────────────────────────────────────────────────────────────────────────────

/// A fixed-capacity ring buffer of past [`Event`]s used for tempo estimation.
pub struct EventHistory {
    events: Vec<Event>,
    max_size: usize,
}

impl EventHistory {
    /// Create an empty history with the given maximum capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            events: Vec::new(),
            max_size,
        }
    }

    /// Push an event, evicting the oldest entry when at capacity.
    pub fn push(&mut self, event: Event) {
        if self.events.len() >= self.max_size {
            self.events.remove(0);
        }
        self.events.push(event);
    }

    /// Mean timing error across all fired events, or `None` if none have fired.
    pub fn average_error(&self) -> Option<f64> {
        let errors: Vec<f64> = self.events.iter().filter_map(|e| e.error()).collect();
        if errors.is_empty() {
            None
        } else {
            Some(errors.iter().sum::<f64>() / errors.len() as f64)
        }
    }

    /// Estimate the current tempo from the timestamps of fired `Beat` events.
    ///
    /// Returns `None` if fewer than two fired beat events are available.
    pub fn tempo_estimate(&self) -> Option<f64> {
        let beats: Vec<f64> = self
            .events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::Beat) && e.actual_at.is_some())
            .filter_map(|e| e.actual_at)
            .collect();

        if beats.len() < 2 {
            return None;
        }

        let first = beats[0];
        let last = beats[beats.len() - 1];
        let n = beats.len() as f64;
        Some(60.0 * (n - 1.0) / (last - first))
    }

    /// Return up to `n` most recent beat events (as references).
    pub fn recent_beats(&self, n: usize) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| matches!(e.kind, EventKind::Beat))
            .rev()
            .take(n)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// predictor
// ─────────────────────────────────────────────────────────────────────────────

/// A local tempo model that predicts when the next beat, bar, or phrase will occur.
pub struct Predictor {
    /// Current tempo estimate in beats-per-minute.
    pub tempo_bpm: f64,
    /// Confidence level in `[0.0, 1.0]`; higher means less drift.
    pub confidence: f64,
}

impl Predictor {
    /// Create a predictor with the given tempo and full confidence.
    pub fn new(tempo_bpm: f64) -> Self {
        Self {
            tempo_bpm,
            confidence: 1.0,
        }
    }

    /// Duration of a single beat in seconds.
    pub fn beat_duration_secs(&self) -> f64 {
        60.0 / self.tempo_bpm
    }

    /// Predict the wall time of the next beat.
    ///
    /// `current_phase` is the within-beat phase `[0.0, 1.0)`.
    pub fn predict_next_beat(&self, current_time: f64, current_phase: f64) -> f64 {
        let time_to_next = (1.0 - current_phase) * self.beat_duration_secs();
        current_time + time_to_next
    }

    /// Predict the wall time of the next bar boundary.
    ///
    /// `bar_phase` is the position within the current bar `[0.0, 1.0)`.
    pub fn predict_next_bar(
        &self,
        current_time: f64,
        bar_phase: f64,
        beats_per_bar: u32,
    ) -> f64 {
        let bar_duration = self.beat_duration_secs() * beats_per_bar as f64;
        let time_to_next = (1.0 - bar_phase) * bar_duration;
        current_time + time_to_next
    }

    /// Predict the wall time of the next phrase boundary.
    ///
    /// `phrase_phase` is the position within the current phrase `[0.0, 1.0)`.
    pub fn predict_next_phrase(
        &self,
        current_time: f64,
        phrase_phase: f64,
        beats_per_phrase: u32,
    ) -> f64 {
        let phrase_duration = self.beat_duration_secs() * beats_per_phrase as f64;
        let time_to_next = (1.0 - phrase_phase) * phrase_duration;
        current_time + time_to_next
    }

    /// Update the tempo model using observed history.
    ///
    /// Blends `0.8 * current + 0.2 * estimated` when a new estimate is available.
    /// Confidence rises when average error is low.
    pub fn update_from_history(&mut self, history: &EventHistory) {
        if let Some(estimated_bpm) = history.tempo_estimate() {
            self.tempo_bpm = 0.8 * self.tempo_bpm + 0.2 * estimated_bpm;
        }
        if let Some(avg_err) = history.average_error() {
            // Map error magnitude to confidence: 0 error -> 1.0, 0.1s error -> ~0.0
            let err_magnitude = avg_err.abs();
            self.confidence = (1.0 - err_magnitude * 10.0).clamp(0.0, 1.0);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// drift
// ─────────────────────────────────────────────────────────────────────────────

/// A pairwise drift measurement between two agents.
#[derive(Debug, Clone)]
pub struct DriftReport {
    /// ID of the first agent.
    pub agent_a: u64,
    /// ID of the second agent.
    pub agent_b: u64,
    /// How far ahead agent A is relative to agent B, in milliseconds.
    /// Positive means agent A is ahead.
    pub drift_ms: f64,
    /// Suggested tempo multiplier for agent A to correct the drift.
    pub correction: f64,
}

impl DriftReport {
    /// Create a drift report and compute the suggested correction factor.
    ///
    /// `correction = (1.0 - drift_ms / 10_000.0).clamp(0.9, 1.1)`
    pub fn new(agent_a: u64, agent_b: u64, drift_ms: f64) -> Self {
        let correction = (1.0 - drift_ms / 10_000.0).clamp(0.9, 1.1);
        Self {
            agent_a,
            agent_b,
            drift_ms,
            correction,
        }
    }

    /// Returns `true` when the absolute drift exceeds 10 ms.
    pub fn is_significant(&self) -> bool {
        self.drift_ms.abs() > 10.0
    }
}

/// Accumulates [`DriftReport`]s and derives ensemble-wide correction guidance.
pub struct DriftDetector {
    /// Maximum tolerated drift before a correction is flagged.
    pub tolerance_ms: f64,
    reports: Vec<DriftReport>,
}

impl DriftDetector {
    /// Create a new detector with the given tolerance.
    pub fn new(tolerance_ms: f64) -> Self {
        Self {
            tolerance_ms,
            reports: Vec::new(),
        }
    }

    /// Record a new drift report.
    pub fn record(&mut self, report: DriftReport) {
        self.reports.push(report);
    }

    /// Mean drift across all recorded reports in milliseconds. Returns `0.0` if empty.
    pub fn average_drift(&self) -> f64 {
        if self.reports.is_empty() {
            return 0.0;
        }
        self.reports.iter().map(|r| r.drift_ms).sum::<f64>() / self.reports.len() as f64
    }

    /// Returns `true` if any recorded report is significant.
    pub fn needs_correction(&self) -> bool {
        self.reports.iter().any(|r| r.is_significant())
    }

    /// Mean correction factor across all reports when correction is needed, otherwise `1.0`.
    pub fn correction_factor(&self) -> f64 {
        if !self.needs_correction() {
            return 1.0;
        }
        self.reports.iter().map(|r| r.correction).sum::<f64>() / self.reports.len() as f64
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// clock
// ─────────────────────────────────────────────────────────────────────────────

/// The top-level local clock for a single agent.
///
/// Agents do not synchronise to a shared master clock. Each `TMinusClock` accumulates
/// phase locally, fires events when boundaries are crossed, and can be nudged via
/// [`sync_to`](TMinusClock::sync_to) when an observed beat timestamp arrives from a peer.
pub struct TMinusClock {
    /// Current tempo in beats-per-minute.
    pub tempo_bpm: f64,
    /// Swing amount in `[0.0, 0.5]`. `0.0` = straight time.
    pub swing: f64,
    /// Phase accumulator.
    pub phase_tracker: PhaseTracker,
    /// Local tempo predictor.
    pub predictor: Predictor,
    /// Ring-buffer history for tempo estimation.
    pub history: EventHistory,
    /// Simulated wall clock (seconds since construction).
    pub wall_time: f64,
    pending_events: Vec<Event>,
}

impl TMinusClock {
    /// Create a new clock at 0.0 s with no swing.
    pub fn new(tempo_bpm: f64, beats_per_bar: u32, bars_per_phrase: u32) -> Self {
        Self {
            tempo_bpm,
            swing: 0.0,
            phase_tracker: PhaseTracker::new(beats_per_bar, bars_per_phrase),
            predictor: Predictor::new(tempo_bpm),
            history: EventHistory::new(64),
            wall_time: 0.0,
            pending_events: Vec::new(),
        }
    }

    /// Advance the simulated clock by `dt` seconds.
    ///
    /// Returns a list of events that fired during this tick.
    pub fn tick(&mut self, dt: f64) -> Vec<Event> {
        self.wall_time += dt;

        let dt_beats = dt * self.tempo_bpm / 60.0;
        let result = self.phase_tracker.advance(dt_beats);

        let mut fired: Vec<Event> = Vec::new();

        // Fire phrase events first (coarsest), then bars, then beats so that a
        // simultaneous boundary gets all three events in one tick.
        for _ in 0..result.phrases_fired {
            let mut ev = Event::new(
                EventKind::Phrase,
                self.wall_time - dt,
                self.wall_time,
            );
            ev.mark_fired(self.wall_time);
            self.history.push(ev.clone());
            fired.push(ev);
        }

        for _ in 0..result.bars_fired {
            let mut ev = Event::new(
                EventKind::Bar,
                self.wall_time - dt,
                self.wall_time,
            );
            ev.mark_fired(self.wall_time);
            self.history.push(ev.clone());
            fired.push(ev);
        }

        for _ in 0..result.beats_fired {
            let mut ev = Event::new(
                EventKind::Beat,
                self.wall_time - dt,
                self.wall_time,
            );
            ev.mark_fired(self.wall_time);
            self.history.push(ev.clone());
            fired.push(ev);
        }

        fired
    }

    /// Compute and add the next predicted beat to the pending queue.
    pub fn schedule_next(&mut self) {
        let next_beat_time = self.predictor.predict_next_beat(
            self.wall_time,
            self.phase_tracker.beat_phase,
        );
        let ev = Event::new(EventKind::Beat, self.wall_time, next_beat_time);
        self.pending_events.push(ev);
    }

    /// Swing offset (seconds) for a given beat index.
    ///
    /// Even beats are pushed forward by `swing * beat_duration`; odd beats are pulled back.
    pub fn swing_offset(&self, beat_index: u32) -> f64 {
        let beat_dur = self.predictor.beat_duration_secs();
        if beat_index.is_multiple_of(2) {
            self.swing * beat_dur
        } else {
            -(self.swing * beat_dur)
        }
    }

    /// Change the tempo, updating both the clock and its predictor.
    pub fn set_tempo(&mut self, bpm: f64) {
        self.tempo_bpm = bpm;
        self.predictor.tempo_bpm = bpm;
    }

    /// Time (seconds) until the next beat, as estimated by the predictor.
    pub fn time_to_next_beat(&self) -> f64 {
        self.predictor
            .predict_next_beat(self.wall_time, self.phase_tracker.beat_phase)
            - self.wall_time
    }

    /// Incorporate an externally observed beat time (from a peer or metronome) and update the
    /// predictor.
    pub fn sync_to(&mut self, observed_beat_time: f64) {
        let mut ev = Event::new(EventKind::Beat, self.wall_time, observed_beat_time);
        ev.mark_fired(observed_beat_time);
        self.history.push(ev);
        self.predictor.update_from_history(&self.history);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Event ────────────────────────────────────────────────────────────────

    #[test]
    fn event_new() {
        let ev = Event::new(EventKind::Beat, 0.0, 0.5);
        assert_eq!(ev.kind, EventKind::Beat);
        assert_eq!(ev.predicted_at, 0.0);
        assert_eq!(ev.scheduled_for, 0.5);
        assert!(ev.actual_at.is_none());
    }

    #[test]
    fn event_error_before_fired_is_none() {
        let ev = Event::new(EventKind::Beat, 0.0, 0.5);
        assert!(ev.error().is_none());
    }

    #[test]
    fn event_mark_fired() {
        let mut ev = Event::new(EventKind::Beat, 0.0, 0.5);
        ev.mark_fired(0.51);
        assert_eq!(ev.actual_at, Some(0.51));
    }

    #[test]
    fn event_error_after_fire() {
        let mut ev = Event::new(EventKind::Beat, 0.0, 0.5);
        ev.mark_fired(0.51);
        let err = ev.error().unwrap();
        assert!((err - 0.01).abs() < 1e-10);
    }

    #[test]
    fn event_custom_kind_clone() {
        let ev = Event::new(EventKind::Custom(42), 1.0, 2.0);
        let ev2 = ev.clone();
        assert_eq!(ev2.kind, EventKind::Custom(42));
    }

    // ── PhaseTracker ─────────────────────────────────────────────────────────

    #[test]
    fn phase_tracker_new() {
        let pt = PhaseTracker::new(4, 8);
        assert_eq!(pt.phase, 0.0);
        assert_eq!(pt.beat_phase, 0.0);
        assert_eq!(pt.beats_per_bar, 4);
        assert_eq!(pt.bars_per_phrase, 8);
    }

    #[test]
    fn phase_tracker_advance_single_beat() {
        let mut pt = PhaseTracker::new(4, 8);
        let res = pt.advance(1.0);
        assert_eq!(res.beats_fired, 1);
        assert_eq!(res.bars_fired, 0);
        assert_eq!(res.phrases_fired, 0);
        assert!((pt.beat_phase - 0.0).abs() < 1e-10);
    }

    #[test]
    fn phase_tracker_advance_crosses_bar() {
        let mut pt = PhaseTracker::new(4, 8);
        let res = pt.advance(4.0); // exactly one bar
        assert_eq!(res.beats_fired, 4);
        assert_eq!(res.bars_fired, 1);
        assert_eq!(res.phrases_fired, 0);
    }

    #[test]
    fn phase_tracker_advance_crosses_phrase() {
        let mut pt = PhaseTracker::new(4, 8);
        let res = pt.advance(32.0); // 4 beats * 8 bars = 32 beats = one phrase
        assert_eq!(res.beats_fired, 32);
        assert_eq!(res.bars_fired, 8);
        assert_eq!(res.phrases_fired, 1);
    }

    #[test]
    fn phase_tracker_beat_in_bar() {
        let mut pt = PhaseTracker::new(4, 8);
        pt.advance(2.5); // 2.5 beats in -> beat 2 (0-indexed)
        assert_eq!(pt.beat_in_bar(), 2);
    }

    #[test]
    fn phase_tracker_bar_phase() {
        let mut pt = PhaseTracker::new(4, 8);
        pt.advance(2.0); // 2 beats into first bar -> bar_phase = 2/4 = 0.5
        let bp = pt.bar_phase();
        assert!((bp - 0.5).abs() < 1e-10);
    }

    #[test]
    fn phase_tracker_phrase_phase() {
        let mut pt = PhaseTracker::new(4, 8);
        pt.advance(16.0); // half a phrase (32 beats total)
        let pp = pt.phrase_phase();
        assert!((pp - 0.5).abs() < 1e-10);
    }

    #[test]
    fn phase_advance_result_counts() {
        let mut pt = PhaseTracker::new(4, 8);
        // Advance 33 beats: 2 phrases (64 beats) -> no; 33 beats = 1 phrase + 1 beat
        let res = pt.advance(33.0);
        assert_eq!(res.phrases_fired, 1);
        assert_eq!(res.bars_fired, 8); // 32 beats = 8 bars, plus 1 extra beat (no extra bar)
        assert_eq!(res.beats_fired, 33);
    }

    // ── EventHistory ─────────────────────────────────────────────────────────

    #[test]
    fn history_new_is_empty() {
        let h = EventHistory::new(10);
        assert!(h.average_error().is_none());
        assert!(h.tempo_estimate().is_none());
    }

    #[test]
    fn history_push_evicts_oldest() {
        let mut h = EventHistory::new(2);
        h.push(Event::new(EventKind::Beat, 0.0, 0.5));
        h.push(Event::new(EventKind::Beat, 0.5, 1.0));
        h.push(Event::new(EventKind::Beat, 1.0, 1.5)); // should evict first
        assert_eq!(h.events.len(), 2);
        assert_eq!(h.events[0].scheduled_for, 1.0);
    }

    #[test]
    fn history_average_error_no_fired() {
        let mut h = EventHistory::new(10);
        h.push(Event::new(EventKind::Beat, 0.0, 0.5)); // not fired
        assert!(h.average_error().is_none());
    }

    #[test]
    fn history_average_error_with_fired() {
        let mut h = EventHistory::new(10);
        let mut ev1 = Event::new(EventKind::Beat, 0.0, 0.5);
        ev1.mark_fired(0.51); // error = +0.01
        let mut ev2 = Event::new(EventKind::Beat, 0.5, 1.0);
        ev2.mark_fired(0.99); // error = -0.01
        h.push(ev1);
        h.push(ev2);
        let avg = h.average_error().unwrap();
        assert!(avg.abs() < 1e-10); // mean of +0.01 and -0.01 = 0
    }

    #[test]
    fn history_tempo_estimate_less_than_two_returns_none() {
        let mut h = EventHistory::new(10);
        let mut ev = Event::new(EventKind::Beat, 0.0, 0.5);
        ev.mark_fired(0.5);
        h.push(ev);
        assert!(h.tempo_estimate().is_none());
    }

    #[test]
    fn history_tempo_estimate_120bpm() {
        let mut h = EventHistory::new(10);
        // 120 BPM -> beat every 0.5 s
        for i in 0..5u64 {
            let t = i as f64 * 0.5;
            let mut ev = Event::new(EventKind::Beat, t, t);
            ev.mark_fired(t);
            h.push(ev);
        }
        let bpm = h.tempo_estimate().unwrap();
        assert!((bpm - 120.0).abs() < 1e-6);
    }

    // ── Predictor ────────────────────────────────────────────────────────────

    #[test]
    fn predictor_next_beat_phase_zero_returns_full_duration() {
        let p = Predictor::new(120.0); // beat = 0.5 s
        let next = p.predict_next_beat(10.0, 0.0);
        assert!((next - 10.5).abs() < 1e-10);
    }

    #[test]
    fn predictor_next_beat_phase_half_returns_half_duration() {
        let p = Predictor::new(120.0);
        let next = p.predict_next_beat(10.0, 0.5);
        assert!((next - 10.25).abs() < 1e-10);
    }

    #[test]
    fn predictor_update_from_history_blends_tempo() {
        let mut p = Predictor::new(120.0);
        let mut h = EventHistory::new(10);
        // Build history implying 100 BPM (beat every 0.6 s)
        for i in 0..5u64 {
            let t = i as f64 * 0.6;
            let mut ev = Event::new(EventKind::Beat, t, t);
            ev.mark_fired(t);
            h.push(ev);
        }
        p.update_from_history(&h);
        // expected: 0.8*120 + 0.2*100 = 96 + 20 = 116
        assert!((p.tempo_bpm - 116.0).abs() < 1e-6);
    }

    // ── DriftReport ──────────────────────────────────────────────────────────

    #[test]
    fn drift_report_new() {
        let r = DriftReport::new(1, 2, 5.0);
        assert_eq!(r.agent_a, 1);
        assert_eq!(r.agent_b, 2);
        assert_eq!(r.drift_ms, 5.0);
    }

    #[test]
    fn drift_report_is_significant_threshold() {
        let below = DriftReport::new(1, 2, 5.0);
        assert!(!below.is_significant());
        let above = DriftReport::new(1, 2, 15.0);
        assert!(above.is_significant());
        let exact = DriftReport::new(1, 2, 10.0);
        assert!(!exact.is_significant()); // strictly greater than 10
    }

    #[test]
    fn drift_report_correction_clamped() {
        // drift = 2000 ms -> 1.0 - 0.2 = 0.8, clamped to 0.9
        let r_low = DriftReport::new(1, 2, 2000.0);
        assert!((r_low.correction - 0.9).abs() < 1e-10);
        // drift = -2000 ms -> 1.0 + 0.2 = 1.2, clamped to 1.1
        let r_high = DriftReport::new(1, 2, -2000.0);
        assert!((r_high.correction - 1.1).abs() < 1e-10);
        // drift = 100 ms -> 1.0 - 0.01 = 0.99 (within range)
        let r_mid = DriftReport::new(1, 2, 100.0);
        assert!((r_mid.correction - 0.99).abs() < 1e-10);
    }

    // ── DriftDetector ────────────────────────────────────────────────────────

    #[test]
    fn drift_detector_record_and_average() {
        let mut d = DriftDetector::new(10.0);
        d.record(DriftReport::new(1, 2, 20.0));
        d.record(DriftReport::new(1, 3, -10.0));
        assert!((d.average_drift() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn drift_detector_needs_correction() {
        let mut d = DriftDetector::new(10.0);
        d.record(DriftReport::new(1, 2, 5.0)); // not significant
        assert!(!d.needs_correction());
        d.record(DriftReport::new(1, 3, 50.0)); // significant
        assert!(d.needs_correction());
    }

    #[test]
    fn drift_detector_correction_factor_no_correction() {
        let d = DriftDetector::new(10.0);
        assert!((d.correction_factor() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn drift_detector_correction_factor_with_reports() {
        let mut d = DriftDetector::new(10.0);
        // 20 ms drift -> correction = 1.0 - 0.002 = 0.998
        d.record(DriftReport::new(1, 2, 20.0));
        let cf = d.correction_factor();
        assert!((cf - 0.998).abs() < 1e-10);
    }

    // ── TMinusClock ──────────────────────────────────────────────────────────

    #[test]
    fn clock_new() {
        let c = TMinusClock::new(120.0, 4, 8);
        assert_eq!(c.tempo_bpm, 120.0);
        assert_eq!(c.swing, 0.0);
        assert_eq!(c.wall_time, 0.0);
    }

    #[test]
    fn clock_tick_fires_beat_at_right_time() {
        let mut c = TMinusClock::new(120.0, 4, 8); // beat = 0.5 s
        let fired = c.tick(0.5);
        assert!(!fired.is_empty());
        let beats: Vec<_> = fired.iter().filter(|e| e.kind == EventKind::Beat).collect();
        assert_eq!(beats.len(), 1);
    }

    #[test]
    fn clock_tick_accumulates_phase() {
        let mut c = TMinusClock::new(120.0, 4, 8);
        c.tick(0.25); // half a beat
        assert!((c.phase_tracker.beat_phase - 0.5).abs() < 1e-10);
    }

    #[test]
    fn clock_swing_offset_sign_alternates() {
        let mut c = TMinusClock::new(120.0, 4, 8);
        c.swing = 0.1;
        let off0 = c.swing_offset(0);
        let off1 = c.swing_offset(1);
        assert!(off0 > 0.0);
        assert!(off1 < 0.0);
        assert!((off0 + off1).abs() < 1e-15);
    }

    #[test]
    fn clock_set_tempo() {
        let mut c = TMinusClock::new(120.0, 4, 8);
        c.set_tempo(140.0);
        assert_eq!(c.tempo_bpm, 140.0);
        assert_eq!(c.predictor.tempo_bpm, 140.0);
    }

    #[test]
    fn clock_time_to_next_beat() {
        let c = TMinusClock::new(120.0, 4, 8); // phase=0, so full beat = 0.5s
        let t = c.time_to_next_beat();
        assert!((t - 0.5).abs() < 1e-10);
    }

    #[test]
    fn clock_multiple_ticks_across_bar_boundary() {
        let mut c = TMinusClock::new(120.0, 4, 8); // bar = 2.0 s
        let mut bar_events = 0;
        // Tick 4 times * 0.5 s = 2.0 s total -> should cross one bar
        for _ in 0..4 {
            let fired = c.tick(0.5);
            bar_events += fired.iter().filter(|e| e.kind == EventKind::Bar).count();
        }
        assert_eq!(bar_events, 1);
    }
}
