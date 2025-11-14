for dir in ./*; do
  (cd "$dir" && rm -f *.mp3; mv converted/* .; rmdir converted;)
done
