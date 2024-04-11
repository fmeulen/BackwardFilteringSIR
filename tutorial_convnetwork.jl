using Statistics
using Flux, Flux.Optimise
using MLDatasets: CIFAR10
using Images.ImageCore
using Flux: onehotbatch, onecold
using Base.Iterators: partition
using CUDA



train_x, train_y = CIFAR10.traindata(Float32)
labels = onehotbatch(train_y, 0:9)