#![forbid(unsafe_code)]

//! T-minus event simulation. Ticks count up; negative = before downbeat.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EventKind {
    Note { pitch: u8, velocity: u8 },
    Sync,
    Trigger { id: u32 },
    Countdown { remaining: i64 },
    Halt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TMinusEvent {
    pub tick: i64,
    pub kind: EventKind,
}

impl TMinusEvent {
    pub fn new(tick: i64, kind: EventKind) -> Self {
        TMinusEvent { tick, kind }
    }
}

#[derive(Debug, Default)]
pub struct Timeline {
    events: Vec<TMinusEvent>,
}

impl Timeline {
    pub fn new() -> Self {
        Timeline { events: Vec::new() }
    }

    pub fn schedule(&mut self, event: TMinusEvent) {
        let pos = self.events.partition_point(|e| e.tick <= event.tick);
        self.events.insert(pos, event);
    }

    pub fn drain_until(&mut self, up_to: i64) -> Vec<TMinusEvent> {
        let split = self.events.partition_point(|e| e.tick <= up_to);
        self.events.drain(..split).collect()
    }

    pub fn len(&self) -> usize { self.events.len() }
    pub fn is_empty(&self) -> bool { self.events.is_empty() }
    pub fn next_tick(&self) -> Option<i64> { self.events.first().map(|e| e.tick) }
}

pub struct Simulator {
    timeline: Timeline,
    pub current_tick: i64,
}

impl Simulator {
    pub fn new(start_tick: i64) -> Self {
        Simulator { timeline: Timeline::new(), current_tick: start_tick }
    }

    pub fn schedule(&mut self, event: TMinusEvent) {
        self.timeline.schedule(event);
    }

    pub fn tick(&mut self) -> Vec<TMinusEvent> {
        self.current_tick += 1;
        self.timeline.drain_until(self.current_tick)
    }

    pub fn advance_to(&mut self, target: i64) -> Vec<TMinusEvent> {
        let mut fired = Vec::new();
        while self.current_tick < target {
            fired.extend(self.tick());
        }
        fired
    }

    pub fn pending(&self) -> usize { self.timeline.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn event_kind_eq() { assert_eq!(EventKind::Sync, EventKind::Sync); }
    #[test] fn event_kind_ne() { assert_ne!(EventKind::Sync, EventKind::Halt); }
    #[test] fn event_kind_note_eq() {
        assert_eq!(EventKind::Note { pitch: 60, velocity: 80 }, EventKind::Note { pitch: 60, velocity: 80 });
    }
    #[test] fn event_kind_note_ne() {
        assert_ne!(EventKind::Note { pitch: 60, velocity: 80 }, EventKind::Note { pitch: 61, velocity: 80 });
    }
    #[test] fn event_kind_trigger_eq() {
        assert_eq!(EventKind::Trigger { id: 42 }, EventKind::Trigger { id: 42 });
    }
    #[test] fn event_kind_countdown() {
        assert_eq!(EventKind::Countdown { remaining: -5 }, EventKind::Countdown { remaining: -5 });
    }
    #[test] fn tminus_event_new() {
        let e = TMinusEvent::new(-10, EventKind::Sync);
        assert_eq!(e.tick, -10);
        assert_eq!(e.kind, EventKind::Sync);
    }
    #[test] fn tminus_event_eq() {
        assert_eq!(TMinusEvent::new(0, EventKind::Halt), TMinusEvent::new(0, EventKind::Halt));
    }
    #[test] fn timeline_empty() {
        let t = Timeline::new();
        assert!(t.is_empty());
        assert_eq!(t.next_tick(), None);
    }
    #[test] fn timeline_schedule_order() {
        let mut t = Timeline::new();
        t.schedule(TMinusEvent::new(5, EventKind::Sync));
        t.schedule(TMinusEvent::new(-3, EventKind::Sync));
        t.schedule(TMinusEvent::new(0, EventKind::Halt));
        assert_eq!(t.next_tick(), Some(-3));
    }
    #[test] fn timeline_drain_until() {
        let mut t = Timeline::new();
        t.schedule(TMinusEvent::new(-3, EventKind::Sync));
        t.schedule(TMinusEvent::new(0, EventKind::Halt));
        t.schedule(TMinusEvent::new(5, EventKind::Sync));
        let fired = t.drain_until(0);
        assert_eq!(fired.len(), 2);
        assert_eq!(fired[0].tick, -3);
        assert_eq!(fired[1].tick, 0);
        assert_eq!(t.len(), 1);
    }
    #[test] fn timeline_drain_until_none() {
        let mut t = Timeline::new();
        t.schedule(TMinusEvent::new(10, EventKind::Sync));
        assert!(t.drain_until(5).is_empty());
    }
    #[test] fn timeline_same_tick_two_events() {
        let mut t = Timeline::new();
        t.schedule(TMinusEvent::new(3, EventKind::Trigger { id: 1 }));
        t.schedule(TMinusEvent::new(3, EventKind::Trigger { id: 2 }));
        assert_eq!(t.drain_until(3).len(), 2);
    }
    #[test] fn simulator_initial_tick() {
        assert_eq!(Simulator::new(-10).current_tick, -10);
    }
    #[test] fn simulator_tick_increments() {
        let mut sim = Simulator::new(-10);
        sim.tick();
        assert_eq!(sim.current_tick, -9);
    }
    #[test] fn simulator_fires_at_tick() {
        let mut sim = Simulator::new(-5);
        sim.schedule(TMinusEvent::new(-4, EventKind::Note { pitch: 60, velocity: 100 }));
        let fired = sim.tick();
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].kind, EventKind::Note { pitch: 60, velocity: 100 });
    }
    #[test] fn simulator_no_early_fire() {
        let mut sim = Simulator::new(-5);
        sim.schedule(TMinusEvent::new(0, EventKind::Halt));
        assert!(sim.tick().is_empty());
    }
    #[test] fn simulator_advance_to() {
        let mut sim = Simulator::new(-10);
        sim.schedule(TMinusEvent::new(-8, EventKind::Sync));
        sim.schedule(TMinusEvent::new(-5, EventKind::Sync));
        sim.schedule(TMinusEvent::new(0, EventKind::Halt));
        let fired = sim.advance_to(0);
        assert_eq!(fired.len(), 3);
        assert_eq!(sim.current_tick, 0);
    }
    #[test] fn simulator_pending_decreases() {
        let mut sim = Simulator::new(-3);
        sim.schedule(TMinusEvent::new(-2, EventKind::Sync));
        sim.schedule(TMinusEvent::new(-1, EventKind::Sync));
        assert_eq!(sim.pending(), 2);
        sim.tick(); sim.tick();
        assert_eq!(sim.pending(), 0);
    }
    #[test] fn simulator_zero_crossing() {
        let mut sim = Simulator::new(-1);
        sim.schedule(TMinusEvent::new(0, EventKind::Note { pitch: 64, velocity: 127 }));
        assert_eq!(sim.tick().len(), 1);
    }
    #[test] fn simulator_advance_to_noop_when_at_target() {
        let mut sim = Simulator::new(5);
        sim.schedule(TMinusEvent::new(10, EventKind::Halt));
        let fired = sim.advance_to(5);
        assert!(fired.is_empty());
        assert_eq!(sim.current_tick, 5);
    }
}
