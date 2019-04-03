package net.davidmcginnis.ml.AnomalyDetection

import org.apache.spark.rdd.RDD

trait AnomalyDetector {
  def predict(points : RDD[Seq[Double]]) : RDD[(Seq[Double], Boolean)]
}
