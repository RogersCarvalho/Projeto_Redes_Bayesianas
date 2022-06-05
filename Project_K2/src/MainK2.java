import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.Random;
import java.util.StringTokenizer;

import weka.classifiers.bayes.BayesNet;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;


public class MainK2 {
	
	// guarda as instancias do conjunto de dados, lidas a partir do arquivo.
	private Instances instances;
	private double gValue;	
	// eh criada a variavel que representa a rede bayesiana.
	private BayesNet net;
	private K2 k2Algorithm;
	private Evaluation eval;
	int[] vetVariable;
	int parentNumber;
		
	static public void main(String args[])
	{
		new MainK2(args);					
	}
	
	public MainK2(String[] args)
	{	
		try
		{
			if(args.length != 3)
			{
				System.out.println("Erro de sintaxe!");
				System.out.println("Sintaxe correta: K2.jar " +
						           "'nome do arquivo da base de dados'.arff " +
						           //"'nome do arquivo da base de testes'.arff " +
						           "'indice da variavel classe' " +
						           //"'nome da classe' " +						      
						           "'numero maximo de pais'");
				return;
			}
			System.out.println("Executando o algoritmo K2...");
			
			// guarda o tempo de inicio da execucao....
			long initialTime = System.currentTimeMillis();
			// armazena as instancias de treinamento.
			instances = readDataSetFile(args[0]);
			// armazena a ordenacao de variaveis a ser seguida.
			//vetVariable = readOrderingFile(args[1]);
			// guarda o indice da variavel classe.
			//instances.setClassIndex(vetVariable[0]);
			instances.setClassIndex(Integer.parseInt(args[1]));
			System.out.println("Classe definida: " + instances.classAttribute());
			parentNumber = Integer.parseInt(args[2]);
			
			builtBayesianNetwork(parentNumber);
			// retorna o valor de g.
			gValue = net.measureBayesScore();
			
			// guarda o tempo de termino da execucao  
			long finalTime = System.currentTimeMillis();
			// Calcula tempo de execucao do algoritmo DMBC.
			double runTime = (double)((finalTime - initialTime));

			// Informacoes da inducao do modelo
			showNet(net.graph(), gValue);
			System.out.println("Tempo para induzir o modelo (ms): " + runTime);
						
			// Salva em arquivo a rede bayesiana aprendida pelo K2.
		    String outputName = "K2_";
		    String outputFileName = args[0].replace(".arff", ".xml");
		    outputFileName = outputName.concat(outputFileName);
		    FileWriter fileWriter = new FileWriter(outputFileName);
			PrintWriter fileNet = new PrintWriter(fileWriter, true);
			fileNet.println(net.graph());
			fileNet.close();
			
			// Salva informacoes da rede bayesiana, como variavel classe, valor da funcao g, 
			// tempo de inducao e numero de vezes que a funcao g foi executada.
			FileWriter fileBN = new FileWriter("Information_BN_K2.txt", true);
			PrintWriter fw = new PrintWriter(fileBN, true);
			fw.println(instances.classAttribute() + "\t" + gValue + "\t" + runTime);
			fw.close();
			
			//crossValidationClassification();
			trainAndTestSplitClassification();
			// Salva os valores de classificacao
			FileWriter fileClassification = new FileWriter("Classification_VOMOS.txt", true);
			PrintWriter fclass = new PrintWriter(fileClassification, true);
			fclass.println(eval.toSummaryString("\nResults\n======\n", false));
			fclass.println(eval.toClassDetailsString());
			fclass.println(eval.toMatrixString(("\n=== Confusion matrix ===\n")));
			fclass.close();
					
			System.out.println("\nExecucao do K2 concluida com sucesso!");
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
		// construtor
		// dela, passamos o caminho do arquivo que queremos ler.
		FileReader arffReader = new FileReader(arffName);

		// Instances é uma classe do Weka, e no construtor dela podemos passar
		// um
		// objeto FileReader (que construímos na linha
		// acima). Daí o Weka recupera os dados do arquivo pelo FileReader e
		// monta
		// uma coleção de Instance num objeto Instances
		Instances instances = new Instances(arffReader);
		arffReader.close(); // fechamos o leitor de arquivo

		return instances; // retorna o objeto de Instances
	}
	
	private int[] readOrderingFile(String orderingFileName)
	{		
		int vet[] = new int[instances.numAttributes()];
				
		// variavel que guarda o arquivo de entrada da populacao inicial.
		File orderingFile = new File(orderingFileName);
		try
		{
			BufferedReader reading = new BufferedReader(new FileReader(
					orderingFile));			
			String line;
			int position=0;
			while((line = reading.readLine()) != null)	
			{
				StringTokenizer st = new StringTokenizer(line);
				while(st.hasMoreTokens())
				{
					vet[position] = (Integer.parseInt(st.nextToken()));
					position++;
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
	
	private void builtBayesianNetwork(int numberOfParent)
	{
		
		try
		{
			k2Algorithm = new K2();
			// eh criada a variavel que representa a rede bayesiana.
			net = new BayesNet();
			// armazena as informacoes da base de dados.
			net.m_Instances = instances;
			// parametros de inicializacao do DMBC:
			// 1 - inicia como naive bayes ou nao:
			k2Algorithm.setInitAsNaiveBayes(false);
			// 2 - utiliza o markov blanket ou nao:
			k2Algorithm.setMarkovBlanketClassifier(false);
			// 3 - configura o numero maximo de pais:
			k2Algorithm.setMaxNrOfParents(numberOfParent);
			// 4 - utiliza a ordenacao inicial ou nao:
			k2Algorithm.setRandomOrder(false);	
			// 5 - utiliza a ordenacao passada pelo usuario.
			//k2Algorithm.setVariableOrdering(vetVariable);
			// inicia o contador de chamadas da funcao g.
			//k2Algorithm.setGFunctionCounter(0);
			// constroi a estrutura da rede e calcula as probabilidades.
			net.setSearchAlgorithm(k2Algorithm);
			// passa ao DMBC a ordenacao de variaveis que sera usada no aprendizado.
			//dmbcAlgorithm.setVariableOrdering(vetOrdering);
			// aprende a estrutura da rede.
			net.initStructure();
			net.buildStructure();
			// calcula os parametros numericos
			net.estimateCPTs();
		}
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void showNet(String bayesianNet, double fitness)
	{		
		try 
		{
			//System.out.println("\n\nRede Bayesiana em XML: ");			
			//System.out.println(bayesianNet);
			System.out.println("\n=====Informacoes da Rede Bayesiana=====");
			System.out.println("Fitness: " + fitness);
		} 
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void crossValidationClassification()
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
				//showNet(netCopy.graph(), netCopy.measureBayesScore(), k2Algorithm.getGFunctionCounter());
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
	
	private void trainClassification()
	{
		// realiza a classificacao usando o conjunto inteiro de dados
		try 
		{
			net.buildClassifier(instances);
			Evaluation evaluation = new Evaluation(instances);
			evaluation.evaluateModel(net,instances);
			System.out.println("\n=== Classification using the training dataset ===");
			System.out.println("=== Sumary ===");
			// How many correct instances?
			double correctValue = evaluation.correct();
			double numberOfInstances = instances.numInstances();
			System.out.println("Correctly Classified Instances: " + correctValue + 
					            "  /  " + ((correctValue*100)/numberOfInstances) + " %");
			// How many incorrect instances?
			double incorrectValue = evaluation.incorrect();
			System.out.println("Incorrectly Classified Instances: " + incorrectValue + 
		            "  /  " + ((incorrectValue*100)/numberOfInstances) + " %");
			// How many instances?
			System.out.println("Total number of instances: " + numberOfInstances);
		}
		catch (Exception e) 
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private void trainAndTestSplitClassification()
	{
		try
		{
			int seed=1;
			Random rnd = new Random(seed);
			instances.randomize(rnd);
			double percent = 66.0;
	
			String[] options;
	
	
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
