package net.davidmcginnis.ml.AnomalyDetection

class LOFPoint(private val _point : Seq[Double]) extends Serializable {
  def getDistanceTo(other : LOFPoint) : Double = {
    Math.sqrt(_point.zip(other._point).map{ case (x, y) => Math.pow(x - y, 2)}.sum)
  }

}
