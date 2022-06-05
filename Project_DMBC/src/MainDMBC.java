import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import java.util.StringTokenizer;

import classifiers.bayes.net.search.local.DMBC;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.gui.beans.Classifier;


public class MainDMBC{
	
	private double gValue;	
	// eh criada a variavel que representa a rede bayesiana.
	private BayesNet net;
	private DMBC dmbcAlgorithm;
	private Evaluation eval;
	int[] vetVariable;
	int classIndex;
	int parentNumber;
		
	static public void main(String args[])
	{
		new MainDMBC(args);					
	}
	
	public void buildBayesianNetwork(Instances instances) throws Exception
	{
		
		// armazena as informacoes da base de dados.
		net.m_Instances = instances;
		// parametros de inicializacao do DMBC:
		// 1 - inicia como naive bayes ou nao:
		dmbcAlgorithm.setInitAsNaiveBayes(false);
		// 2 - utiliza o markov blanket ou nao:
		dmbcAlgorithm.setMarkovBlanketClassifier(false);
		// 3 - configura o numero maximo de pais:
		dmbcAlgorithm.setMaxNrOfParents(parentNumber);
		// 4 - utiliza a ordenacao inicial ou nao:
		dmbcAlgorithm.setRandomOrder(false);	
		// 5 - utiliza a ordenacao passada pelo usuario.
		//dmbcAlgorithm.setVariableOrdering(vetVariable);
		// constroi a estrutura da rede e calcula as probabilidades.
		net.setSearchAlgorithm(dmbcAlgorithm);
		// passa ao DMBC a ordenacao de variaveis que sera usada no aprendizado.
		//dmbcAlgorithm.setVariableOrdering(vetOrdering);
		// aprende a estrutura da rede.
		net.initStructure();
		net.buildStructure();
		// calcula os parametros numericos
		net.estimateCPTs();
		// retorna o valor de g.
		gValue = net.measureBayesScore();
	}

	public double classifyInstance(Instance instance) throws Exception
	{
		return net.classifyInstance(instance);
	}

	
	public MainDMBC(String[] args)
	{	
		try
		{
			// guarda as instancias do conjunto de dados, lidas a partir do arquivo.
			Instances instances;
			net = new BayesNet();
			dmbcAlgorithm = new DMBC();
			
			if(args.length != 3)
			{
				System.out.println("Erro de sintaxe!");
				System.out.println("Sintaxe correta: DMBC.jar " +
						           "'nome do arquivo da base de dados'.arff " +
						           //"'nome do arquivo da base de testes'.arff " +
						           "'indice da variavel classe' " +
						           //"'nome da classe' " +						      
						           "'numero maximo de pais'");
				return;
			}
			System.out.println("Executando o algoritmo DMBC...");
			
			// guarda o tempo de inicio da execucao....
			long initialTime = System.currentTimeMillis();
			// armazena as instancias de treinamento.
			instances = readDataSetFile(args[0]);
			// armazena a ordenacao de variaveis a ser seguida.
			//vetVariable = readOrderingFile(args[1], instances);
			//classIndex = vetVariable[0];
			//instances.setClassIndex(classIndex);
			instances.setClassIndex(Integer.parseInt(args[1]));
			System.out.println("Classe definida: " + instances.classAttribute());
			parentNumber = Integer.parseInt(args[2]);
			
			// constroi o classificador dmbc
			buildBayesianNetwork(instances);		
					
			// guarda o tempo de termino da execucao  
			long finalTime = System.currentTimeMillis();
			// Calcula tempo de execucao do algoritmo DMBC.
			double runTime = (double)((finalTime - initialTime)/1000);

			showNet(net.graph(), gValue);
			System.out.println("Tempo de execucao: " + runTime);
			
			// Salva em arquivo a rede bayesiana aprendida pelo DMBC.
		    String outputName = "DMBC_";
		    String outputFileName = args[0].replace(".arff", ".xml");
		    outputFileName = outputName.concat(outputFileName);
		    FileWriter fileWriter = new FileWriter(outputFileName);
			PrintWriter fileNet = new PrintWriter(fileWriter, true);
			fileNet.println(net.graph());
			fileNet.close();
			
			// Salva informacoes da rede bayesiana, como variavel classe, valor da funcao g, 
			// tempo de inducao.
			FileWriter fileBN = new FileWriter("Information_BN_DMBC.txt", true);
			PrintWriter fw = new PrintWriter(fileBN, true);
			fw.println(instances.classAttribute() + "\t" + gValue + "\t" + runTime);
			fw.close();
			
			//crossValidationClassification(instances);
			trainAndTestSplitClassification(instances);
			// Salva os valores de classificacao
			FileWriter fileClassification = new FileWriter("Classification_DMBC.txt", true);
			PrintWriter fclass = new PrintWriter(fileClassification, true);
			fclass.println(eval.toSummaryString("\nResults\n======\n", false));
			fclass.println(eval.toClassDetailsString());
			fclass.println(eval.toMatrixString(("\n=== Confusion matrix ===\n")));
			fclass.close();
						
			System.out.println("\nExecucao do DMBC concluida com sucesso!");
			
			/*** realizando a avaliacao do classificador *****/
			//instances = readDataSetFile(args[3]); // le os dados de testes.
			//instances.setClassIndex(classIndex);
			//net.m_Instances = instances;
			//Evaluation evaluation = new Evaluation(instances);
			//evaluation.evaluateModel(net, instances);
			// How many correct instances?
			//System.out.print(evaluation.correct()+"/");
			// How many instances?
			//System.out.println(instances.numInstances());
			
						
		}
		catch (Exception e) 
		{
			e.printStackTrace();
		}		
	}

	private Instances readDataSetFile(String dataSetFileName) 
	{
		Instances instances = null;
		try 
		{
			System.out.println("Lendo o arquivo da base de dados: " + dataSetFileName);
			instances = getInstances(dataSetFileName);
		} 
		catch (Exception e) 
		{
			e.printStackTrace();
			System.out.println("A leitura do arquivo " + dataSetFileName + 
	                           " nao foi possivel!");
		}
		return instances;
	}
	
	/**
	 * @param arffName
	 *            - O caminho completo do arquivo Arff
	 * @return - Um objeto Instances (do Weka) que é uma coleção de vários
	 *         objetos Instance (cada Instance representa uma linha do arquivo
	 *         ARFF)
	 * @throws Exception
	 *             - se algo der errado, lança uma Exceçao genérica
	 */
	private static Instances getInstances(String arffName) throws Exception {
		// FileReader é uma classe do Java para leitura de arquivos. No
		// construtor dela, passamos o caminho do arquivo que queremos ler.
		FileReader arffReader = new FileReader(arffName);

		// Instances é uma classe do Weka, e no construtor dela podemos passar
		// um
		// objeto FileReader (que construímos na linha
		// acima). Daí o Weka recupera os dados do arquivo pelo FileReader e
		// monta uma coleção de Instance num objeto Instances
		Instances instances = new Instances(arffReader);
		arffReader.close(); // fechamos o leitor de arquivo

		return instances; // retorna o objeto de Instances
	}
	
	private int[] readOrderingFile(String orderingFileName, Instances instances)
	{		
		int vet[] = new int[instances.numAttributes()];
				
		// variavel que guarda o arquivo de entrada da populacao inicial.
		File orderingFile = new File(orderingFileName);
		try
		{
			BufferedReader reading = new BufferedReader(new FileReader(
					orderingFile));			
			String line;
			int variable;
			int position=0;
			
			initVetPosition(vet);
			//vet[position] = classIndex;
			//position++;
			while((line = reading.readLine()) != null)	
			{
				StringTokenizer st = new StringTokenizer(line);
				while(st.hasMoreTokens())
				{
					variable = (Integer.parseInt(st.nextToken()));
					if( !(isPresent(vet, variable)) )
					{
						vet[position] = variable;
						position++;
					}
				}		
			}
			reading.close();
		}
		catch(Exception e)
		{	
			e.printStackTrace();
			System.out.println("A leitura do arquivo " + orderingFileName + 
					           " nao foi possivel!");
		}
		return vet;
	}
	
	private void initVetPosition(int vet[])
	{
		for(int i=0; i<vet.length; i++)
		{
			vet[i] = -1;
		}
	}
	
	private boolean isPresent(int vet[], int variable)
	{
		for(int i=0; i<vet.length; i++)
		{
			if(vet[i] == variable)
				return true;
		}
		return false;
	}
	
	private void showNet(String bayesianNet, double fitness)
	{		
		try 
		{
			System.out.println("\n=====Informacoes da Rede Bayesiana=====");
			//System.out.println("\n\nRede Bayesiana em XML: ");			
			//System.out.println(bayesianNet);
			System.out.println("Fitness: " + fitness);
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void crossValidationClassification(Instances instances)
	{
		// realiza a classifacao usando a estrategia de validacao cruzada (cross-validation)
		try 
		{
			int seed  = 1;
		    int folds = 10;
		    
		    // randomize data
		    Random rand = new Random(seed);
		    Instances randData = new Instances(instances);
		    randData.randomize(rand);
		    if (randData.classAttribute().isNominal())
		      randData.stratify(folds);
		    
		    // perform cross-validation
		    eval = new Evaluation(randData);
		    for (int n = 0; n < folds; n++) 
		    {
		    	System.out.println("\n === Fold: " + n + " ====");
		    	Instances train = randData.trainCV(folds, n, rand);
				Instances test = randData.testCV(folds, n);
				// the above code is used by the StratifiedRemoveFolds filter, the
				// code below by the Explorer/Experimenter:
				// Instances train = randData.trainCV(folds, n, rand);
				
				// build and evaluate classifier
				BayesNet netCopy = (BayesNet) BayesNet.makeCopy(net);
				// Classifier clsCopy = Classifier.makeCopy(cls);
				netCopy.m_Instances = train;
				netCopy.initStructure();
				netCopy.buildStructure();
				netCopy.estimateCPTs();
				//showNet(netCopy.graph(), netCopy.measureBayesScore());
				netCopy.buildClassifier(train);
				eval.evaluateModel(netCopy, test);
		    }
		    
		    // output evaluation
		    System.out.println();
		    System.out.println("=== Setup ===");
		    System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
		}
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void trainAndTestSplitClassification(Instances instances)
	{
		try
		{
			int seed=1;
			Random rnd = new Random(seed);
			instances.randomize(rnd);
			double percent = 66.0;
	
			System.out.println("\nPerforming " + percent +"% split evaluation");
	
			int trainSize = (int) Math.round(instances.numInstances()*percent/100);
	
			int testSize = instances.numInstances()-trainSize;
	
			Instances train = new Instances (instances, 0, trainSize);
			Instances test = new Instances (instances, trainSize,testSize);
			
			// build and evaluate classifier
			BayesNet netCopy = (BayesNet) BayesNet.makeCopy(net);
			netCopy.m_Instances = train;
			netCopy.initStructure();
			netCopy.buildStructure();
			netCopy.estimateCPTs();
			//showNet(netCopy.graph(), netCopy.measureBayesScore(), k2Algorithm.getGFunctionCounter());
			netCopy.buildClassifier(train);
			eval = new Evaluation(train);
			eval.evaluateModel(netCopy, test);
			
			System.out.println(eval.toSummaryString("\nResults\n======\n", false));
			System.out.println(eval.toClassDetailsString());
			System.out.println(eval.toMatrixString(("\n=== Confusion matrix ===\n")));
		}
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
}
