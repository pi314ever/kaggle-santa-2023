# Must be in same directory as train.py
# 
# Usage: ./train.sh

puzzles = ("cube_3", "cube_4", "cube_5", "cube_6", "cube_7", "cube_8", "cube_9", "cube_10", "cube_19", "cube_33", "wreath_21", "wreath_33", "wreath_100", "globe_1_8", "globe_1_16", "globe_2_6", "globe_3_4", "globe_6_4", "globe_6_8", "globe_6_10", "globe_3_33", "globe_8_25")

# Cube 3
python train.py configs/cube_3.yml ../models/cube_3_resnet --device cuda --num_workers 0

# Cube 4
python train.py configs/cube_4.yml ../models/cube_4_resnet --device cuda --num_workers 0

# Cube 5
python train.py configs/cube_5.yml ../models/cube_5_resnet --device cuda --num_workers 0

# Cube 6
python train.py configs/cube_6.yml ../models/cube_6_resnet --device cuda --num_workers 0

# Cube 7
python train.py configs/cube_7.yml ../models/cube_7_resnet --device cuda --num_workers 0

# Cube 8
python train.py configs/cube_8.yml ../models/cube_8_resnet --device cuda --num_workers 0

# Cube 9
python train.py configs/cube_9.yml ../models/cube_9_resnet --device cuda --num_workers 0

# Cube 10
python train.py configs/cube_10.yml ../models/cube_10_resnet --device cuda --num_workers 0

# Cube 19
python train.py configs/cube_19.yml ../models/cube_19_resnet --device cuda --num_workers 0

# Cube 33
python train.py configs/cube_33.yml ../models/cube_33_resnet --device cuda --num_workers 0

# Wreath 21
python train.py configs/wreath_21.yml ../models/wreath_21_resnet --device cuda --num_workers 0

# Wreath 33
python train.py configs/wreath_33.yml ../models/wreath_33_resnet --device cuda --num_workers 0

# Wreath 100
python train.py configs/wreath_100.yml ../models/wreath_100_resnet --device cuda --num_workers 0

# Globe 1/8
python train.py configs/globe_1_8.yml ../models/globe_1_8_resnet --device cuda --num_workers 0

# Globe 1/16
python train.py configs/globe_1_16.yml ../models/globe_1_16_resnet --device cuda --num_workers 0

# Globe 2/6
python train.py configs/globe_2_6.yml ../models/globe_2_6_resnet --device cuda --num_workers 0

# Globe 3/4
python train.py configs/globe_3_4.yml ../models/globe_3_4_resnet --device cuda --num_workers 0

# Globe 6/4
python train.py configs/globe_6_4.yml ../models/globe_6_4_resnet --device cuda --num_workers 0

# Globe 6/8
python train.py configs/globe_6_8.yml ../models/globe_6_8_resnet --device cuda --num_workers 0

# Globe 6/10
python train.py configs/globe_6_10.yml ../models/globe_6_10_resnet --device cuda --num_workers 0

# Globe 3/33
python train.py configs/globe_3_33.yml ../models/globe_3_33_resnet --device cuda --num_workers 0

# Globe 8/25
python train.py configs/globe_8_25.yml ../models/globe_8_25_resnet --device cuda --num_workers 0
