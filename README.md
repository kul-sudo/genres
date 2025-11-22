1. Choose
   - `mode = "Train"`
   - `mode = "Test"`
2. Run `cargo run --release`

| Path  | Purpose |
|-------|-----|
| `checkpoint/model*.mpk` | Best model |
| `./model.mpk` | Last model |

Do not forget to add the `MODEL` var before testing. Make sure your test files have the sample rate of 44100 and are (ideally) stored in ogg.
