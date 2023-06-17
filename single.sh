# learning rate
# sh train.sh SingleNet 18 4 0.01 2 true
# sh train.sh SingleNet 18 4 0.05 2 true
# sh train.sh SingleNet 18 4 0.1 2 true

# sh train.sh SingleNet 18 4 0.01 2 false
# sh train.sh SingleNet 18 4 0.05 2 false
# sh train.sh SingleNet 18 4 0.1 2 false

# sh train.sh SingleNet 34 4 0.01 2 true
sh train.sh SingleNet 34 4 0.05 2 true
sh train.sh SingleNet 34 4 0.1 2 true

sh train.sh SingleNet 34 4 0.01 2 false
sh train.sh SingleNet 34 4 0.05 2 false
sh train.sh SingleNet 34 4 0.1 2 false

## batch size
sh train.sh SingleNet 18 8 0.01 2 true
sh train.sh SingleNet 18 8 0.01 2 false

sh train.sh SingleNet 18 16 0.01 2 true
sh train.sh SingleNet 18 16 0.01 2 false

sh train.sh SingleNet 34 8 0.01 2 true
sh train.sh SingleNet 34 8 0.01 2 false

sh train.sh SingleNet 34 16 0.01 2 true
sh train.sh SingleNet 34 16 0.01 2 false