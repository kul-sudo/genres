for dir in ./*; do
  (cd "$dir" && mkdir -p converted && for file in *; do ffmpeg -i "$file" -ar 44100 -ac 1 -map_metadata -1 "converted/${file%.*}.ogg"; done)
done
