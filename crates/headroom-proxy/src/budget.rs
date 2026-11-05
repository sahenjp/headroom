//! USD spend budget per period, computed from the savings ledger events.

use headroom_store::ledger::SavingsEvent;

/// Budget period.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPeriod {
    Hourly,
    Daily,
    Monthly,
}

impl BudgetPeriod {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "hourly" => Some(Self::Hourly),
            "daily" => Some(Self::Daily),
            "monthly" => Some(Self::Monthly),
            _ => None,
        }
    }

    /// Window start (ms) for a period given now (ms).
    fn window_start(&self, now_ms: i64) -> i64 {
        match self {
            Self::Hourly => now_ms - 3_600_000,
            Self::Daily => {
                let day = now_ms.div_euclid(86_400_000);
                day * 86_400_000
            }
            Self::Monthly => {
                // Approximate: 30-day window anchored at the UTC month start
                // would need calendar math; use the last 30 days rolling.
                now_ms - 30 * 86_400_000
            }
        }
    }
}

impl std::fmt::Display for BudgetPeriod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Hourly => "hourly",
            Self::Daily => "daily",
            Self::Monthly => "monthly",
        };
        f.write_str(s)
    }
}

/// Budget definition.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    pub limit_usd: f64,
    pub period: BudgetPeriod,
}

/// Decision after checking the ledger against the budget.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BudgetDecision {
    Within { spent_usd: f64 },
    Exceeded { spent_usd: f64, limit_usd: f64 },
}

/// Sum cost_usd of events inside the period window.
pub fn spent_since(events: &[SavingsEvent], period: &BudgetPeriod, now_ms: i64) -> f64 {
    let start = period.window_start(now_ms);
    events
        .iter()
        .filter(|e| {
            parse_ts_ms(&e.ts)
                .map(|ts| ts >= start && ts <= now_ms)
                .unwrap_or(false)
        })
        .map(|e| e.cost_usd)
        .sum()
}

/// Check the budget against ledger events.
pub fn check(events: &[SavingsEvent], budget: &Budget, now_ms: i64) -> BudgetDecision {
    let spent = spent_since(events, &budget.period, now_ms);
    if spent > budget.limit_usd {
        BudgetDecision::Exceeded { spent_usd: spent, limit_usd: budget.limit_usd }
    } else {
        BudgetDecision::Within { spent_usd: spent }
    }
}

/// ISO-8601 "YYYY-MM-DDTHH:MM:SS(.fff)Z" -> ms (mirrors ledger.rs).
fn parse_ts_ms(ts: &str) -> Option<i64> {
    let y: i64 = ts.get(0..4)?.parse().ok()?;
    let mo: i64 = ts.get(5..7)?.parse().ok()?;
    let d: i64 = ts.get(8..10)?.parse().ok()?;
    let h: i64 = ts.get(11..13)?.parse().ok()?;
    let mi: i64 = ts.get(14..16)?.parse().ok()?;
    let s: i64 = ts.get(17..19)?.parse().ok()?;
    Some(days_from_civil(y, mo as u32, d as u32) * 86_400_000 + (h * 3600 + mi * 60 + s) * 1000)
}

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = if m > 2 { m - 3 } else { m + 9 } as i64;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev_at(ts: &str, cost: f64) -> SavingsEvent {
        SavingsEvent::new_proxy("m", "c", 100, 50).tap_cost(cost).tap_ts(ts)
    }

    trait Tap {
        fn tap_cost(self, cost: f64) -> SavingsEvent;
        fn tap_ts(self, ts: &str) -> SavingsEvent;
    }

    impl Tap for SavingsEvent {
        fn tap_cost(mut self, cost: f64) -> SavingsEvent {
            self.cost_usd = cost;
            self
        }
        fn tap_ts(mut self, ts: &str) -> SavingsEvent {
            self.ts = ts.to_string();
            self
        }
    }

    #[test]
    fn parse_period() {
        assert_eq!(BudgetPeriod::parse("daily"), Some(BudgetPeriod::Daily));
        assert_eq!(BudgetPeriod::parse("HOURLY"), Some(BudgetPeriod::Hourly));
        assert_eq!(BudgetPeriod::parse("bogus"), None);
    }

    #[test]
    fn hourly_window_math() {
        // now = 2026-11-05T12:00:20Z; hourly window starts 11:00:20Z.
        let now = 1_793_880_020_000i64;
        let in_window = ev_at("2026-11-05T11:30:00Z", 1.0);
        let out = ev_at("2026-11-05T10:00:00Z", 100.0);
        let spent = spent_since(&[in_window, out], &BudgetPeriod::Hourly, now);
        assert!((spent - 1.0).abs() < 1e-9, "spent={spent}");
    }

    #[test]
    fn budget_check_exceeded() {
        let now = 1_793_880_020_000i64;
        let events = vec![ev_at("2026-11-05T11:50:00Z", 5.0)];
        let budget = Budget { limit_usd: 3.0, period: BudgetPeriod::Daily };
        match check(&events, &budget, now) {
            BudgetDecision::Exceeded { spent_usd, limit_usd } => {
                assert!((spent_usd - 5.0).abs() < 1e-9);
                assert!((limit_usd - 3.0).abs() < 1e-9);
            }
            BudgetDecision::Within { .. } => panic!("should exceed"),
        }
    }
}
