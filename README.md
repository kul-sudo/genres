1. Choose
   - `mode = "Train"`
   - `mode = "Test"`
2. Run `cargo run --release`

| Path  | Purpose |
|-------|-----|
| `checkpoint/model*.mpk` | Best model |
| `./model.mpk` | Last model |

Do not forget to add the `MODEL` var before testing.
Make sure your test files have the sample rate of 44100 and are (ideally) stored in ogg: `for f in *; do [[ -f "$f" ]] && [[ "$f" != *.ogg ]] && ffmpeg -i "$f" -vn -ar 44100 -acodec libvorbis "${f%.*}.ogg" && rm "$f"; done`
