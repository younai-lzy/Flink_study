package com.sina;

import com.alibaba.alink.operator.batch.BatchOperator;
import com.alibaba.alink.operator.batch.dataproc.SplitBatchOp;
import com.alibaba.alink.operator.batch.evaluation.EvalRegressionBatchOp;
import com.alibaba.alink.operator.batch.source.CsvSourceBatchOp;
import com.alibaba.alink.operator.common.evaluation.RegressionMetrics;
import com.alibaba.alink.operator.common.evaluation.TuningRegressionMetric;
import com.alibaba.alink.pipeline.PipelineModel;
import com.alibaba.alink.pipeline.classification.RandomForestClassifier;
import com.alibaba.alink.pipeline.regression.DecisionTreeRegressionModel;
import com.alibaba.alink.pipeline.regression.DecisionTreeRegressor;
import com.alibaba.alink.pipeline.regression.LinearRegression;
import com.alibaba.alink.pipeline.regression.LinearRegressionModel;
import com.alibaba.alink.pipeline.tuning.*;

public class AlinkTest {
  public static void main(String[] args) throws Exception {

    //设置并行度为1
    BatchOperator.setParallelism(1);
    //读取数据
    String filePath = "input/白血病数据集.txt";
    String schema
      = "Obs double, age double, sex double, bmi double, map double, tc double, ldl double, hdl double, tch double, ltg double, glu double, y double";
    CsvSourceBatchOp csvSource = new CsvSourceBatchOp()
      .setFilePath(filePath)
      .setSchemaStr(schema)
      .setIgnoreFirstLine(true) //设置首行忽略
      .setFieldDelimiter(" ");
    csvSource.print();

    //创建Maven Project添加pom依赖，创建Flink程序，调用Alink API，设置并行度为1，读取白血病数据集，
    // 提取特征features和目标值label，明确目标值label对应的样本字段，并注明业务含义，输出控制台；（5分）
    String[] features = {"Obs", "age", "sex", "bmi", "map", "tc", "ldl", "hdl", "tch", "ltg", "glu"};
    String label = "y";

    //（2）、调用Alink API，按照7/2/1比例划分数据集为：
    // 训练数据集trainData、验证集和测试数据集testData，统计数据条目数，输出控制台；（5分）
    //拆分数据
    BatchOperator <?> spliter = new SplitBatchOp().setFraction(0.7);
    BatchOperator<?> trainData = spliter.linkFrom(csvSource);
    BatchOperator<?> otherData = spliter.getSideOutput(0);

    BatchOperator <?> spliter2 = new SplitBatchOp().setFraction(2.0/3);
    BatchOperator<?> validationData = spliter2.linkFrom(otherData);
    BatchOperator<?> testData = spliter2.getSideOutput(0);

    //统计数据条目数，输出控制台
    System.out.println("trainData的条目数：" + trainData.count());
    System.out.println("testData的条目数：" + testData.count());
    System.out.println("validationData的条目数：" + validationData.count());


    //（3）、构建算法模型：从线性回归相关算法、决策树算法中，任选2种算法，
    // 使用训练数据trainData，分别构建不同算法模型，合理设置参数值；（5分）

    //创建线性回归模型 超参数设置不同的值（每个算法至少2个超参数值设置
    LinearRegression lr = new LinearRegression()
      .setFeatureCols(features)
      .setLabelCol(label)
      .setMaxIter(100)
      .setLearningRate(0.1)
      .setL1(0.1)
      .setPredictionCol("pred")
      .enableLazyPrintModelInfo();

    //训练
    LinearRegressionModel model1 = lr.fit(trainData);
    //预测
    BatchOperator<?> lrResult = model1.transform(validationData);
    lrResult.print();

    //决策树回归模型  超参数设置不同的值（每个算法至少2个超参数值设置
    DecisionTreeRegressor tree = new DecisionTreeRegressor()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .setMaxDepth(10)
      .setMaxBins(128)
      .setMinSamplesPerLeaf(1)
      .setFeatureCols(features)
      .enableLazyPrintModelInfo();

    //训练
    DecisionTreeRegressionModel model2 = tree.fit(trainData);
    //预测验证数据集
    BatchOperator<?> treeResult = model2.transform(validationData);
    treeResult.print();

    //评估
    //线性回归的评估 至少输出2个评估指标
    RegressionMetrics lrmetrics = new EvalRegressionBatchOp()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .linkFrom(lrResult)
      .collectMetrics();

    System.out.println("线性回归的RMSE:" + lrmetrics.getRmse());
    System.out.println("线性回归的MSE:" + lrmetrics.getMse());


    // 决策树回归的评估 至少输出2个评估指标
    RegressionMetrics treemetrics = new EvalRegressionBatchOp()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .linkFrom(treeResult)
      .collectMetrics();

    System.out.println("决策树回归的RMSE:" + treemetrics.getRmse());
    System.out.println("决策树回归的MSE:" + treemetrics.getMse());

    //线性回归的网格搜索      训练预测评估模型，获取最佳模型
    ParamGrid paramGrid1 = new ParamGrid()
      .addGrid(lr, LinearRegression.MAX_ITER, new Integer[] {100,200})
      .addGrid(lr, LinearRegression.LEARNING_RATE, new Double[] {0.1,0.2});

    //设置回归评估器
    RegressionTuningEvaluator lrTuningEvaluator = new RegressionTuningEvaluator()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .setTuningRegressionMetric(TuningRegressionMetric.MSE);

    //网格搜索
    GridSearchCV cv1 = new GridSearchCV()
      .setEstimator(lr)
      .setParamGrid(paramGrid1)
      .setTuningEvaluator(lrTuningEvaluator)
      .setNumFolds(2)
      .enableLazyPrintTrainInfo("TrainInfo");
    GridSearchCVModel lrModel = cv1.fit(csvSource);
    //得到最佳模型
    PipelineModel lrBestPipelineModel = lrModel.getBestPipelineModel();
    BatchOperator<?> lrBestResult = lrBestPipelineModel.transform(testData);

    //评估
    RegressionMetrics lrMetrics = new EvalRegressionBatchOp()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .linkFrom(lrBestResult)
      .collectMetrics();

    //并计算评估指标MSE
    System.out.println("线性回归最佳模型的testData的MSE:" + lrMetrics.getMse());



    //决策树回归的网格搜索
    ParamGrid paramGrid2 = new ParamGrid()
      .addGrid(tree, DecisionTreeRegressor.MAX_BINS, new Integer[] {128,256})
      .addGrid(tree, DecisionTreeRegressor.MAX_DEPTH, new Integer[] {10,20});

    //设置回归评估器
    RegressionTuningEvaluator treeTuningEvaluator = new RegressionTuningEvaluator()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .setTuningRegressionMetric(TuningRegressionMetric.MSE);

    //设置网格搜索
    GridSearchCV cv2 = new GridSearchCV()
      .setEstimator(tree)
      .setParamGrid(paramGrid2)
      .setTuningEvaluator(treeTuningEvaluator)
      .setNumFolds(2)
      .enableLazyPrintTrainInfo("TrainInfo");
    GridSearchCVModel treeModel = cv2.fit(csvSource);
    //得到最佳模型
    PipelineModel treeBestPipelineModel = treeModel.getBestPipelineModel();
    BatchOperator<?> treeBestResult = treeBestPipelineModel.transform(testData);

    //评估
    RegressionMetrics treeMetrics = new EvalRegressionBatchOp()
      .setPredictionCol("pred")
      .setLabelCol(label)
      .linkFrom(treeBestResult)
      .collectMetrics();
    //打印
    //并计算评估指标MSE
    System.out.println("决策树回归最佳模型的testData的MSE:" + treeMetrics.getMse());

    //（7）、算法模型保存：获取每个算法算法的最佳模型，并保存模型到本地文件系统；
    //保存
    lrBestPipelineModel.save("output/lrBestModel.model",true);
    treeBestPipelineModel.save("output/treeBestModel.model", true);

    BatchOperator.execute();
  }
}
