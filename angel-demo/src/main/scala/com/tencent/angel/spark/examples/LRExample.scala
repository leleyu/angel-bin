package com.tencent.angel.spark.examples

import com.tencent.angel.ml.core.conf.MLConf
import com.tencent.angel.ml.core.network.layers.edge.inputlayer.DenseInputLayer
import com.tencent.angel.ml.core.network.layers.edge.inputlayer.SparseInputLayer
import com.tencent.angel.ml.core.network.layers.edge.losslayer.SimpleLossLayer
import com.tencent.angel.ml.core.network.transfunc.Identity
import com.tencent.angel.ml.core.optimizer.Adam
import com.tencent.angel.ml.core.optimizer.Momentum
import com.tencent.angel.ml.core.optimizer.SGD
import com.tencent.angel.ml.core.optimizer.loss.LogLoss
import com.tencent.angel.spark.ml.core.GraphModel

class LRExample extends GraphModel {

  val lr = conf.getDouble(MLConf.ML_LEARN_RATE)

  override
  def network(): Unit = {
    // inputlayer

    // losslayer
    
  }
}