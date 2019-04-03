package net.davidmcginnis.ml.AnomalyDetection

import util.Random.nextGaussian
import breeze.plot._
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD

object AnomalyDetectionMain extends App {
  // Algorithm from https://micvog.com/2016/05/21/using-spark-for-anomaly-fraud-detection/
  override def main(args: Array[String]) : Unit = {
    val config = new SparkConf()
    val context = new SparkContext("local", "Anomaly Detection Prototype", config)
    val points = generatePoints(context, 10000, 0.8)

    val variance1 = new VarianceAnomalyDetector(points._1, 2.0)
    plotData(variance1.predict(points._2).cache(), "Variance Method, Epsilon = 2.0")

    val variance2 = new VarianceAnomalyDetector(points._1, 5.0)
    plotData(variance2.predict(points._2).cache(), "Variance Method, Epsilon = 5.0")

    val variance3 = new VarianceAnomalyDetector(points._1, 7.0)
    plotData(variance3.predict(points._2).cache(), "Variance Method, Epsilon = 7.0")

    val variance4 = new VarianceAnomalyDetector(points._1, 8.0)
    plotData(variance4.predict(points._2).cache(), "Variance Method, Epsilon = 8.0")
  }

  def generatePoints(context : SparkContext, n : Integer, percentTrain : Double) : (RDD[Seq[Double]], RDD[Seq[Double]]) = {
    val determinePoint = () => {
      Seq.apply(nextGaussian(), nextGaussian())
    }
    val numTrain = (percentTrain * n).toInt
    val trainRange = (1 to n).take(numTrain)
    val numTest = (n * (1 - percentTrain)).toInt
    val testRange = (1 to n).take(numTest)
    val trainPoints = context.makeRDD(trainRange.map(_ => determinePoint()))
    val testPoints = context.makeRDD(testRange.map(_ => determinePoint()))
    (trainPoints, testPoints)
  }

  def plotData(points : RDD[(Seq[Double], Boolean)], title : String) : Unit = {
    val insides = points.filter(_._2).collect()
    val insideX = insides.map(_._1.head).toSeq
    val insideY = insides.map(_._1.last).toSeq

    val outsides = points.filter(!_._2).collect()
    val outsideX = outsides.map(_._1.head).toSeq
    val outsideY = outsides.map(_._1.last).toSeq

    val fig = Figure(title)
    val plt = fig.subplot(0)
    plt += plot(insideX, insideY, '.')
    plt += plot(outsideX, outsideY, '.')
    fig.refresh()
  }
}
