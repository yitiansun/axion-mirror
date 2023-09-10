# Axion mirror

![](outputs/graveyard_samples.gif)

## Run status
|                 |  src  | CHIME | CHORD | HERA  | HIRAX256 | HIRAX1024 | BURSTT256 | BURSTT2048 |
|-----------------|-------|-------|-------|-------|----------|-----------|-----------|------------|
| egrs (A1)       |   -   |   X   |   X   |   X   |    X     |     X     |     X     |     X      |
| gsr  (A1)       |   -   |   X   |   X   |   X   |    X     |     X     |     X     |     X      |
| snrf (A1)       |   X   |   X   |   X   |   X   |    X     |           |     X     |            |
| snrp (A1)       |   X   |   X   |   X   |   X   |    X     |           |     X     |            |
| snrg (A1)       |   X   |   X   |   X   |   X   |    X     |           |     X     |            |
| snro (A2)       |   -   |   X   |   X   |   X   |    X     |           |     X     |            |
| reach base (B1) |   -   |   X   |   X   |   X   |    X     |           |     X     |            |
| reach var (B2)  |   -   |   X   |   X   |   X   |    X     |           |           |            |
| snrfnf (A3)     |       |       |       |       |          |           |           |            |
| reach fnf (A3)  |       |       |       |       |          |           |           |            |

- B: base
- E: ti - time index / electron model
- T: tf - t_free
- R: sr - snr rate

- f = fullinfo : BET
- p = partialinfo : BET
- o = observed = f+p : BET
- g = graveyard : BETR
- fnf = fullinfo no free expansion