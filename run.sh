#!/bin/bash
bin/angel-example com.tencent.angel.spark.examples.local.NetworkExample network:LogisticRegression input:data/census/census_148d_train.dummy
#bin/angel-example com.tencent.angel.spark.examples.local.NetworkExample network:FactorizationMachine input:data/census/census_148d_train.dummy
#bin/angel-example com.tencent.angel.spark.examples.local.NetworkExample network:DeepFM input:data/census/census_148d_train.dummy ml.epoch.num:100
