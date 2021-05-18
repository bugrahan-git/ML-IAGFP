#!/bin/zsh

movements=($(ls -d ./dataset/*))
for movement in "${movements[@]}"; do
  artists=($(ls -d $movement/*))
  for artist in "${artists[@]}"; do
    images=($(ls -d $artist/*))
    counter=0
      for img in "${images[@]}"; do
        extension=$(echo ${img} | awk -F'.' '{print $NF}')
        newFileName=$(printf "%s%04d.%s" "${artist}" "${counter}" "${extension}")
        mv ${img} ${newFileName}
        echo "${newFileName}"
        counter=$((counter + 1))
      done
  done
done