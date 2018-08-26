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

  val numField: Int = SharedConf.get().getInt(MLConf.ML_FIELD_NUM)
  val numFactor: Int = SharedConf.get().getInt(MLConf.ML_RANK_NUM)
  val lr: Double = SharedConf.get().getDouble(MLConf.ML_LEARN_RATE)

  override
  def network(): Unit = {
    // first order, input layer
    val first = new SparseInputLayer("first", 1, new Identity(), new Adam(lr))
    // embedding
    val embedding = new Embedding("embedding", numField * numFactor, numFactor, new Adam(lr))
    // second order, cross operations
    val second = new BiInnerSumCross("second", embedding)
    // higher order, FC1

    // higher order, FC2

    // higher order, output

    // sum, first + second + higher
    // val sum = new SumPooling("sum", 1, Array[Layer](first, second, higher))
    // losslayer, logloss
    // new SimpleLossLayer("loss", sum, new LogLoss)

  }
}
