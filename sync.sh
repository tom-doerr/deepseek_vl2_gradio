#!/bin/bash
rsync -avz --progress . conic:~/${PWD##*/}/
