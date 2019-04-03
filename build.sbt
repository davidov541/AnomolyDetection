name := "AnomalyDetection"

version := "0.1"

scalaVersion := "2.10.6"

libraryDependencies  ++= Seq(
  // Last stable release
  "org.scalanlp" %% "breeze" % "0.13.2",

  // Native libraries are not included by default. add this if you want them (as of 0.7)
  // Native libraries greatly improve performance, but increase jar sizes.
  // It also packages various blas implementations, which have licenses that may or may not
  // be compatible with the Apache License. No GPL code, as best I know.
  "org.scalanlp" %% "breeze-natives" % "0.13.2",

  // The visualization library is distributed separately as well.
  // It depends on LGPL code
  "org.scalanlp" %% "breeze-viz" % "0.13.2",

  "org.apache.spark" % "spark-core_2.10" % "1.6.3",
)

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Maven Central Server" at "http://repo1.maven.org/maven2",
  "Spark Packages Repo" at "https://dl.bintray.com/spark-packages/maven/",
  "Cloudera" at "https://repository.cloudera.com/artifactory/cloudera-repos",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/",
)
        