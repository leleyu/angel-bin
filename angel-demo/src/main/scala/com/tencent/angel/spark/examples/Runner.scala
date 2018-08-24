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

  def main(args: Array[String]): Unit = {
    val params = ArgsUtil.parse(args)
    val input = params.getOrElse("input", "data/census/census_148d_train.dummy")
    val dataType = params.getOrElse(MLConf.ML_DATA_INPUT_FORMAT, "dummy")
    val features = params.getOrElse(MLConf.ML_FEATURE_INDEX_RANGE, "148").toInt
    val numField = params.getOrElse(MLConf.ML_FIELD_NUM, "13").toInt
    val numRank = params.getOrElse(MLConf.ML_RANK_NUM, "8").toInt
    val numEpoch = params.getOrElse(MLConf.ML_EPOCH_NUM, "200").toInt
    val fraction = params.getOrElse(MLConf.ML_BATCH_SAMPLE_RATIO, "0.1").toDouble
    val lr = params.getOrElse(MLConf.ML_LEARN_RATE, "0.02").toDouble

    val network = params.getOrElse("network", "LogisticRegression")

    SharedConf.addMap(params)
    SharedConf.get().set(MLConf.ML_DATA_INPUT_FORMAT, dataType)
    SharedConf.get().setInt(MLConf.ML_FEATURE_INDEX_RANGE, features)
    SharedConf.get().setInt(MLConf.ML_FIELD_NUM, numField)
    SharedConf.get().setInt(MLConf.ML_RANK_NUM, numRank)
    SharedConf.get().setInt(MLConf.ML_EPOCH_NUM, numEpoch)
    SharedConf.get().setDouble(MLConf.ML_BATCH_SAMPLE_RATIO, fraction)
    SharedConf.get().setDouble(MLConf.ML_LEARN_RATE, lr)


    val className = "com.tencent.angel.spark.examples." + network
    val model = GraphModel(className)
    val learner = new OfflineLearner()

    // load data
    val conf = new SparkConf()
    conf.setMaster("local")
    conf.setAppName(s"$network Example")
    conf.set("spark.ps.model", "LOCAL")
    conf.set("spark.ps.instances", "1")
    conf.set("spark.ps.cores", "1")
    conf.set("spark.ui.enabled", "false")

    val sc = new SparkContext(conf)
    val parser = DataParser(SharedConf.get())
    val data = sc.textFile(input).map(f => parser.parse(f))

    PSContext.getOrCreate(sc)

    def evaluate(data: RDD[LabeledData], bModel: Broadcast[GraphModel]): (Double, Double) = {
      val scores = data.mapPartitions { case iter =>
        val model = bModel.value
        val output = model.forward(iter.toArray)
        Iterator.single((output, model.graph.placeHolder.getLabel))
      }
      (new AUC().cal(scores), new Precision().cal(scores))
    }

    def train(data: RDD[LabeledData], model: GraphModel): Unit = {
      // split data into train and validat

      data.cache()

      // build network
      model.init(data.getNumPartitions)

      val bModel = SparkContext.getOrCreate().broadcast(model)

      var (start, end) = (0L, 0L)
      var (reduceTime, updateTime) = (0L, 0L)

      for (iteration <- 1 to numEpoch) {
        start = System.currentTimeMillis()
        val (lossSum, batchSize) = data.sample(false, fraction, 42 + iteration)
          .mapPartitions { case iter =>
            PSClient.instance()
            val samples = iter.toArray
            bModel.value.forward(samples)
            val loss = bModel.value.getLoss()
            bModel.value.backward()
            Iterator.single((loss, samples.size))
          }.reduce((f1, f2) => (f1._1 + f2._1, f1._2 + f2._2))
        end = System.currentTimeMillis()
        reduceTime = end - start

        start = System.currentTimeMillis()
        model.update(iteration)
        end = System.currentTimeMillis()
        updateTime = end - start

        val loss = lossSum / model.graph.taskNum

        println(s"batch[$iteration] batchSize=$batchSize trainLoss=$loss reduceTime=$reduceTime updateTime=$updateTime")

        val numEvaluate = SparkContext.getOrCreate().getConf.getInt("spark.offline.evaluate", 50)

        if (iteration % numEvaluate == 0) {
          val (trainAuc, trainPrecision) = evaluate(data, bModel)
          val trainMetricLog = s"trainAuc=$trainAuc trainPrecision=$trainPrecision"
          println(s"batch[$iteration] $trainMetricLog")
        }
      }
    }

    train(data, model)
    PSContext.stop()
    sc.stop()
  }
}
