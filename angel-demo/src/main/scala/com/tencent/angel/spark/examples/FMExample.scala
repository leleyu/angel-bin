package com.tencent.angel.spark.examples

import com.tencent.angel.ml.core.conf.MLConf
import com.tencent.angel.ml.core.network.layers.edge.inputlayer.{Embedding, SparseInputLayer}
import com.tencent.angel.ml.core.network.layers.edge.losslayer.SimpleLossLayer
import com.tencent.angel.ml.core.network.layers.join.SumPooling
import com.tencent.angel.ml.core.network.layers.linear.BiInnerSumCross
import com.tencent.angel.ml.core.network.transfunc.Identity
import com.tencent.angel.ml.core.optimizer.Adam
import com.tencent.angel.ml.core.optimizer.loss.LogLoss
import com.tencent.angel.spark.ml.core.GraphModel


class FMExample extends GraphModel {

  val lr = conf.getDouble(MLConf.ML_LEARN_RATE)
  val numField = conf.getInt(MLConf.ML_FIELD_NUM)
  val numFactor = conf.getInt(MLConf.ML_RANK_NUM)

  override
  def network(): Unit = {
    // first order, inputlayer
    val first = new SparseInputLayer("first", 1, new Identity(), new Adam(lr))
    // embedding 

    // second order, cross operations

    // sum first and second

    // losslayer
    // new SimpleLossLayer("loss", sum, new LogLoss)
  }
}