package com.tencent.angel.spark.examples

import com.tencent.angel.ml.core.conf.{MLConf, SharedConf}
import com.tencent.angel.ml.core.utils.DataParser
import com.tencent.angel.ml.feature.LabeledData
import com.tencent.angel.spark.client.PSClient
import com.tencent.angel.spark.context.PSContext
import com.tencent.angel.spark.ml.core.metric.{AUC, Precision}
import com.tencent.angel.spark.ml.core.{ArgsUtil, GraphModel, OfflineLearner}
import org.apache.log4j.PropertyConfigurator
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Runner {

  PropertyConfigurator.configure("conf/log4j.properties")

  def evaluate(data: RDD[LabeledData], bModel: Broadcast[GraphModel]): (Double, Double) = {
    val scores = data.mapPartitions { case iter =>
      val model = bModel.value
      val output = model.forward(iter.toArray)
      Iterator.single((output, model.graph.placeHolder.getLabel))
    }
    (new AUC().cal(scores), new Precision().cal(scores))
  }

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "data/census/census_148d_train.dummy")
    val dataType = params.getOrElse(MLConf.ML_DATA_INPUT_FORMAT, "dummy")
    val features = params.getOrElse(MLConf.ML_FEATURE_INDEX_RANGE, "148").toInt
    val numField = params.getOrElse(MLConf.ML_FIELD_NUM, "13").toInt
    val numRank = params.getOrElse(MLConf.ML_RANK_NUM, "5").toInt
    val numEpoch = params.getOrElse(MLConf.ML_EPOCH_NUM, "100").toInt
    val fraction = params.getOrElse(MLConf.ML_BATCH_SAMPLE_RATIO, "0.1").toDouble
    val lr = params.getOrElse(MLConf.ML_LEARN_RATE, "0.02").toDouble

    val network = params.getOrElse("network", "LogisticRegression")

    SharedConf.addMap(params)
    SharedConf.get().set(MLConf.ML_DATA_INPUT_FORMAT, dataType)
    SharedConf.get().setInt(MLConf.ML_FEATURE_INDEX_RANGE, features)
    SharedConf.get().setInt(MLConf.ML_FIELD_NUM, numField)
    SharedConf.get().setInt(MLConf.ML_RANK_NUM, numRank)
    SharedConf.get().setDouble(MLConf.ML_LEARN_RATE, lr)

    val className = "com.tencent.angel.spark.examples." + network
    val model = GraphModel(className)

    // set Spark conf
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName(s"$network Example")
    // set PS conf
    conf.set("spark.ps.model", "LOCAL")
    conf.set("spark.ps.instances", "1")
    conf.set("spark.ps.cores", "1")

    conf.set("spark.ui.enabled", "false")

    // loading data with Spark RDD
    val sc = new SparkContext(conf)
    val parser = DataParser(SharedConf.get())
    val data = sc.textFile(input).map(f => parser.parse(f))

    // launch PS
    PSContext.getOrCreate(sc)

    // cache data into memory
    data.cache()
    // build network
    model.init(data.getNumPartitions)

    // broadcast model
    val bModel = sc.broadcast(model)

    for (iteration <- 1 to numEpoch) {
      val (lossSum, batchSize) = data.sample(false, fraction, 42 + iteration)
        .mapPartitions { case iter =>
          // build connections with servers for each executor
          PSClient.instance()
          val samples = iter.toArray
          bModel.value.forward(samples)
          val loss = bModel.value.getLoss()
          bModel.value.backward()
          Iterator.single((loss, samples.size))
        }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))

      // update model
      model.update(iteration)

      val loss = lossSum / data.getNumPartitions
      println(s"batch[$iteration] batchSize=$batchSize trainLoss=$loss")
    }

    val (auc, precision) = evaluate(data, bModel)
    println(s"trainAuc=$auc trainPrecision=$precision")

    PSContext.stop()
    sc.stop()
  }
}
