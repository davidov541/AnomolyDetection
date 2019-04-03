package net.davidmcginnis.ml.AnomalyDetection
import org.apache.spark.rdd.RDD

class LOFAnomalyDetector(private val _k : Int = 10) extends AnomalyDetector with Serializable {

  override def predict(data: RDD[Seq[Double]]) : RDD[(Seq[Double], Boolean)] = {
    val kNearest = getKNearest(data.map(p => new LOFPoint(p)))
    data.map((_, true))
  }

  private def getKNearest(data: RDD[LOFPoint]) : RDD[(LOFPoint, List[(LOFPoint, List[(LOFPoint, Double)])])] = {
    val distances = data.cartesian(data).flatMap{ case (x, y) =>
      val distance = x.getDistanceTo(y)
      Seq.apply((x, List.apply((y, distance))), (y, List.apply((x, distance))))
    }
    val nearestNeighbors = distances.reduceByKey{ case (l1, l2) =>
      (l1 ++ l2).sortBy{ case (_, d) => d}.take(_k)
    }

    val allNearestNeighbors = nearestNeighbors.flatMap{ case (point, neighbors) => neighbors.map{ case (neighbor, distance) => (neighbor, (point, distance))}}
    val joined = allNearestNeighbors.join(nearestNeighbors).map{ case (neighbor, ((point, _), neighbors)) => (point, List.apply((neighbor, neighbors)))}
    joined.reduceByKey{ case (l1, l2) => l1 ++ l2 }
  }

  private def getLOF(A : (LOFPoint, Seq[(LOFPoint, Double)]), fullData : RDD[(LOFPoint, Seq[(LOFPoint, Double)])]) : Double = {
    val LRDA = getLRD(A, fullData)
    val bPointsRDD = fullData.context.makeRDD(A._2)
    val fullBData = fullData.join(bPointsRDD).mapValues(_._1)
    val LRDB = fullBData.map(B => getLRD(B, fullData))
    val numerator = LRDB.map(lrd => lrd / LRDA).sum()
    val denominator = LRDB.count()
    numerator / denominator
  }

  private def getLRD(A : (LOFPoint, Seq[(LOFPoint, Double)]), fullData : RDD[(LOFPoint, Seq[(LOFPoint, Double)])]) : Double = {
    val kNearestRDD = fullData.context.makeRDD(A._2)
    val reachabilityDistances = fullData.join(kNearestRDD).mapValues(_._1)
    val numerator = reachabilityDistances.map(B => getReachabilityDistance(A._1, B)).sum()
    val denominator = reachabilityDistances.count()
    denominator / numerator
  }

  private def getReachabilityDistance(A : LOFPoint, B : (LOFPoint, Seq[(LOFPoint, Double)])) : Double = {
    Math.max(B._2.minBy(-1 * _._2)._2, A.getDistanceTo(B._1))
  }
}
