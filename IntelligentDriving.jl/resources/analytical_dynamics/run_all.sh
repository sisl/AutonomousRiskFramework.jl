#!zsh

for f in *.mx; do
  maxima -b $f
done

for f in output/*.txt; do
  if [[ ! $f == *_jl.txt ]]; then
    julia fortran2julia.jl $f > ${f%.txt}"_jl.txt"
  else
    rm $f
  fi
done
