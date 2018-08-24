package com.tencent.angel.spark.examples

import com.tencent.angel.ml.core.conf.{MLConf, SharedConf}
import com.tencent.angel.ml.core.network.layers.Layer
import com.tencent.angel.ml.core.network.layers.edge.inputlayer.{Embedding, SparseInputLayer}
import com.tencent.angel.ml.core.network.layers.edge.losslayer.SimpleLossLayer
import com.tencent.angel.ml.core.network.layers.join.SumPooling
import com.tencent.angel.ml.core.network.layers.linear.{BiInnerSumCross, FCLayer}
import com.tencent.angel.ml.core.network.transfunc.{Identity, Relu}
import com.tencent.angel.ml.core.optimizer.Adam
import com.tencent.angel.ml.core.optimizer.loss.LogLoss
import com.tencent.angel.spark.ml.core.GraphModel

class DeepFMExample extends GraphModel {

  val numFields: Int = SharedConf.get().getInt(MLConf.ML_FIELD_NUM)
  val numFactors: Int = SharedConf.get().getInt(MLConf.ML_RANK_NUM)
  val lr: Double = SharedConf.get().getDouble(MLConf.ML_LEARN_RATE)

  override
  def network(): Unit = {
    val wide = new SparseInputLayer("input", 1, new Identity(), new Adam(lr))
    val embedding = new Embedding("embedding", numFields*numFactors, numFactors, new Adam(lr))
    val innerSumCross = new BiInnerSumCross("innerSumPooling", embedding)
    val hidden1 = new FCLayer("hidden1", 80, embedding, new Relu, new Adam(lr))
    val hidden2 = new FCLayer("hidden2", 50, hidden1, new Relu, new Adam(lr))
    val mlpLayer = new FCLayer("hidden3", 1, hidden2, new Identity, new Adam(lr))
    val join = new SumPooling("sumPooling", 1, Array[Layer](wide, innerSumCross, mlpLayer))
    new SimpleLossLayer("simpleLossLayer", join, new LogLoss)
  }
}
