# Memo: Week 6 games at risk of underselling

**To:** Ticketing team
**Date:** October 3, 2026
**Re:** Home games most at risk this week (Oct 6-10), and what moves demand

**Bottom line.** Two of this week's 53 home games are forecast at least 5
points of capacity below their school's normal crowd:

- **Virginia Tech at California:** about 4,600 seats short
- **Nevada at UTEP:** about 4,500 seats short

Three Power-conference games are on a watch list. Two of them are being
played on a Friday night, which is the most reliable way to lose a crowd.

## Most at risk this week

| Game | Kickoff | Forecast vs. normal | Seats short | Why |
|---|---|---|---:|---|
| **Virginia Tech at California** | Sat 12:30 PM PT | 49% vs. 56% of capacity | ~4,600 (~$345k*) | Cross-country conference opponent with little draw; early kickoff |
| **Nevada at UTEP** | Sat 5:00 PM MT | 22% vs. 31% | ~4,500 (~$112k*) | Home team 1-4; low-draw opponent |
| Iowa at Washington | Fri 6:00 PM PT | 93% vs. 96% | ~2,500 | Friday night; home team 3-2 |
| Florida State at Louisville | Fri 7:00 PM ET | 81% vs. 84% | ~2,000 | Friday night; home team 2-3 |
| Illinois at Michigan State | Sat 3:30 PM ET | 91% vs. 94% | ~2,000 | Home team 2-3, well below its usual record |

"Normal" is the school's average fill last season. Forecasts come from
the gradient-boosted model, and the reasons from the fixed-effects
regression. Cal's band is wide (43-80%), so its outcome is the least
certain on the list. Washington's TV slot wasn't listed yet when the
forecast was made, so a national TV window could close part of its gap.
Four games still had no kickoff time and weren't forecast (South Carolina
at Florida, Texas A&M at Missouri, UAB at Memphis, James Madison at
Georgia Southern).

**Suggested actions.** Concentrate this week's promotional budget on Cal
and UTEP: group, youth and alumni-chapter offers, which fill seats
without discounting the season-ticket base. For the Friday-night games,
lean on student and local walk-up messaging, since weeknights lose
commuters first.

## What moves demand (ten seasons, 8,148 home games)

Effects are in points of stadium capacity, comparing each school with
itself. One point is about 515 seats at an average stadium.

- **Weeknight vs. Saturday: -5.0** (about -2,600 seats). This is the
  biggest scheduling lever.
- **Kickoff under 40 F: -6.3.** Wind of 15+ mph: -3.0. Rain: -1 to -3.
- **Noon vs. night kickoff: -2.3.** Afternoon vs. night: -1.0.
- **Power-conference opponent: +7.4.** Ranked opponent: +2.7. FCS
  opponent: -1.8.
- **Lopsided spreads don't empty seats.** No measurable effect on
  attendance. Don't discount just because a game looks like a blowout.

## Pricing headroom at sellout schools

A sellout hides how many more people wanted in. A censored (Tobit) model
estimates that demand at Alabama, Ohio State and Clemson sellouts averages
about 118-120% of capacity. Clemson's 2025 opener against LSU drew an
estimated 131%, roughly 25,000 unmet ticket requests. Marquee games at
these schools are the strongest candidates for higher or dynamic pricing.

## How much to trust this

- **Track record on 2025, a season the model never saw:** average miss of
  6.2 points of capacity (about 2,500 seats). Of the games it flagged as
  at risk, 77% really did come in 5+ points low.
- **What "attendance" means:** the figures are schools' announced
  attendance, which counts tickets distributed. This memo is about ticket
  demand, not bodies in seats.
- **Committed in advance:** this week's forecasts were committed to git on
  Oct 3, before any kickoff. Results will be scored once attendance is
  posted.

\* Dollar figures assume $75 per ticket at power-conference schools and
$25 elsewhere. Actual revenue per seat isn't public.
