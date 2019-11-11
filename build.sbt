import Dependencies._
import sbtassembly.AssemblyPlugin.defaultUniversalScript

ThisBuild / scalaVersion := "2.12.10"

lazy val root = (project in file("."))
  .settings(
    name:="WI-Project",
    libraryDependencies ++= Seq(
      spark,
      sparkSql,
      sparkMLlib
    )
  )

//Generate executable
assemblyOption in assembly := (assemblyOption in assembly).value
  .copy(prependShellScript = Some(defaultUniversalScript(shebang = false)))

//Export to root dir
assemblyOutputPath in assembly := file(baseDirectory.value.getAbsolutePath+"/clickPredict")