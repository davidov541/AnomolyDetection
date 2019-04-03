package net.davidmcginnis.ml.AnomalyDetection
import org.apache.spark.rdd.RDD

class VarianceAnomalyDetector(trainData : RDD[Seq[Double]], private val _epsilon : Double = 2.0) extends AnomalyDetector with Serializable {
  private val _model : Seq[(Double, Double)] = train(trainData)

  private def train(trainData : RDD[Seq[Double]]) : Seq[(Double, Double)] = {
    val sums = sumDimensions(trainData)
    val count = trainData.count()
    val means = sums.map(_ / count)
    val variances = sumDimensions(trainData.map(s => s.zip(means).map{ case (x, u) => Math.pow(x - u, 2) / count}))
    means.zip(variances)
  }

  private def sumDimensions(points : RDD[Seq[Double]]) : Seq[Double] = {
    points.reduce{ case (x, y) => x.zip(y).map{ case (a, b) => a + b }}
  }

  def predict(testData : RDD[Seq[Double]]) : RDD[(Seq[Double], Boolean)] = {
    val probabilities : RDD[(Seq[Double], Double)] = testData.map(s => {
      val dimProbs : Seq[Double] = s.zip(_model).map{case (x : Double, (u : Double, v : Double)) => calculateProbability(x, u, v)}
      val totalProb : Double = dimProbs.product
      val condensedProb : (Seq[Double], Double) = (s, totalProb)
      condensedProb
    })
    probabilities.map{ case (s, p) => (s, p < _epsilon)}
  }

  private def calculateProbability(x : Double, u : Double, v : Double) : Double = {
    val exp = Math.pow(x - u, 2) / (2 * Math.pow(v, 2))
    val denom = Math.sqrt(2 * Math.PI * Math.pow(v, 2))
    Math.exp(exp) / denom
  }
}
