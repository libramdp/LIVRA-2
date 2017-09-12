// CLASIFICADOR DE BAYES
// en base a:
//  https://github.com/4m1g0/openCV-tutorial/tree/master/19-Clasificador%20de%20bayer

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace std;
using namespace cv;

// Clasificacion
#define NCLASE1 9   // para la clase 1 tornillo, pasaremos una imagen de entrenamiento con 9 objetos
#define NCLASE2 7   // para la clase 2 tuerca, pasaremos una imagen de entrenamiento con 7 objetos
#define NCLASE3 9   // para la clase No-tornillo, pasaremos una imagen de entrenamiento con 9 objetos
#define NCLASE4 5   // para la clase No-Tuerca, pasaremos una imagen de entrenamiento con 6 objetos
#define NENTRE  (NCLASE1+NCLASE2+NCLASE3+NCLASE4)  // cantidad total de objetos que participan del entrenamiento

static Ptr<ml::TrainData>
//Clase que encapsula datos de entrenamiento. Esta clase sólo especifica la interfaz de datos de formación, pero no la implementación.

prepare_train_data(const Mat&data, const Mat& responses, int ntrain_samples)

// funcion para dar forma a los datos de entrenamiento
{
	Mat sample_idx = Mat::zeros(1, data.rows, CV_8U);
	Mat train_samples = sample_idx.colRange(0, ntrain_samples);
	train_samples.setTo(Scalar::all(1));

	int nvars = data.cols;
	Mat var_type(nvars + 1, 1, CV_8U);
	var_type.setTo(Scalar::all(ml::VAR_ORDERED));
	var_type.at<uchar>(nvars) = ml::VAR_CATEGORICAL;

	return ml::TrainData::create(data, ml::ROW_SAMPLE, responses, noArray(), sample_idx, noArray(), var_type);
}

int main() {

	Mat imagen, imagen_bin, imagen_neg, imagen_color, dst1, dst2, dst3, dst4, kernel;
	vector<vector<Point>> contours;

	int contornos = 0;
	int i = 0;
	int clase;
	int k = 0;
	double perimetro, area;
	char *nombre_clases[4] = { "Tornillo", "Tuerca" ,"No-tornillo", "No-tuerca"}; 

	// ENTRENAMIENTO

	char  *imagenes[4] = { "Uno.jpg", "Dos.jpg" , "Tres.jpg" ,"Cuatro.jpg"}; 

	// Creacion del clasificador de Bayes

	Ptr<ml::NormalBayesClassifier> modelo_bayes = ml::NormalBayesClassifier::create();

	// Creacion de las formas de matrices para la fase de entrenamiento

	Mat train_data(NENTRE, 2, CV_32FC1); // tantas filas como objetos de entrenamiento y dos columnas para
	// almacenar el par de valores: perimetro,area de todos los objetos.
	
	Mat clase_data(NENTRE, 1, CV_32FC1); // tantas filas como objetos de entrenamiento, con una columna para almacenar
	// la clase de cada uno.

// PREPARACION DE LOS DATOS DE ENTRENAMIENTO -------------------------------------------------------------

// Lazo de procesamiento de cada imagen de prueba y extraccion de datos para 
// completar las matrices [train_data] y [clase_data]

		for (clase = 0; clase < 4; ++clase) 
// carga de la imagen		
		{
		  imagen = imread(imagenes[clase]);
		  if (!imagen.data) {
		  cout << "Error al cargar la imagen" << endl;
		  system("PAUSE");
		  exit(1);
		                   }
// Umbralizar la imagen antes de detectar los contornos	
// Se crea una matriz para alojar la imagen umbralizada
		imagen_bin = Mat(imagen.size(), CV_8UC1, 1);
		// Pasamos la imagen a escalas de gris
		cvtColor(imagen, imagen, CV_BGR2GRAY);
		// umbralizamos
		threshold(imagen, imagen_bin, 200, 255, CV_THRESH_BINARY);
		imagen_neg = Mat(imagen.size(), CV_8UC1, 1);
		bitwise_not(imagen_bin, imagen_neg); //

		kernel = cv::getStructuringElement(MORPH_RECT, Size(5, 5)); 

		dilate(imagen_neg, dst1, kernel, Point(-1, -1), 5); // aplicamos 5 dilataciones 
		erode(dst1, dst2, kernel, Point(-1, -1), 5); // aplicamos 5 erociones para recuperar el tamaño original

// La imagen umbralizada quedo en dst2 !!!!!!!!!!!!!!!!!!!!!													
// Encontramos los contornos

		findContours(dst2, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

// Se crea una matriz para alojar la imagen de los contornos
		Mat contourImagen(dst2.size(), CV_8UC3, Scalar(255, 255, 255));
		Scalar colors1 = Scalar(0, 0,0);
		
		for (int idx = 0; idx < contours.size(); idx++)
		
		drawContours(contourImagen, contours, idx, colors1);

		namedWindow("Contornos imagen de entrenamiento", CV_WINDOW_AUTOSIZE);
		imshow("Contornos imagen de entrenamiento", contourImagen);
		cvWaitKey(0);
		
// Para cada contorno se calcula perimetro y area y se guarda en la matriz [train_data}
// simultanemente con el mismo indice de contorno se guarda la clase en [clase_data]

		for (size_t idx = 0; idx < contours.size(); idx++) 
		{
			area = contourArea(contours[idx], false);
			perimetro = arcLength(contours[idx], 1);

			train_data.at<float>(k, 0) = perimetro;

			train_data.at<float>(k, 1) = area;
			clase_data.at<float>(k) = clase;
			k++;
		}
// Para cada imagen se encuentran una cantidad de contornos
// El numero total calculado es:	
		contornos = contornos + contours.size(); // cada clase entrenada genera su cantidad de contornos
	   }
// Uno de los argumentos de la funcion 	"prepare_train_data' es: ntrain_samples
		int ntrain_samples = contornos;

// Se muestran los datos  obtenidos antes de proceder con el entrenamiento

	cout << " El numero de piezas para entrenamiento es:" << ntrain_samples << endl;
	cout << " La Matriz [train_data] contiene:" << endl;
	cout <<  train_data << endl;
	cout << " La Matriz [clase_data] indica la clase de cada fila de [train_data]," << endl;
// se muestra la traspuesta de [clase_data] para facilitar la visualizacion	
	Mat C = clase_data.t(); // C es la traspuesta de clase_data 
	
	cout << C << endl;
	cout << " corresponde 0 para tornillo" << endl;
	cout << " corresponde 1 para tuerca" << endl;
	cout << " corresponde 2 para No-tornillo" << endl;
	cout << " corresponde 3 para No-tuerca" << endl;
	
// FIN de la obtencion de caracteristicas discriminantes de los objetos

//////////////////////////////////////////////////////////////////////////////////////////////////////////

//  Preparacion de datos
			
	Ptr<ml::TrainData> datos_entrenamiento = prepare_train_data(train_data, clase_data, ntrain_samples);

	// train_data: matriz con filas para pares perimetro/area de todos los objetos de entrenamiento
	// response_data: matriz con filas para el numero de clase de todos los objetos de entrenamiento
	// ntrain_samples: cantidad de muestras de entrenamiento

// ENTRENAMIENTO -
		modelo_bayes->train(datos_entrenamiento);

// FINALIZO EL ENTRENAMIENTO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	// Se crean las matrices necesarias para el procesamiento de la imagen a clasificar

	Mat clasificar, clasificar_bin, clasificar_neg, clasificar_morf, dilata;

	char NombreImagen[] = "engaño.jpg";//"clasificar-new.jpg"; ////

	clasificar = imread(NombreImagen);
	if (!clasificar.data) {
		cout << "Error al cargar la imagen para clasificar objetos" << endl;
		system("PAUSE");
		exit(1);
	}

	clasificar_bin = Mat(clasificar.size(), 8, 1);
	cvtColor(clasificar, clasificar, CV_BGR2GRAY);

	threshold(clasificar, clasificar_bin, 200, 255, CV_THRESH_BINARY);
	bitwise_not(clasificar_bin, clasificar_neg);

	dilate(clasificar_neg, dilata, kernel, Point(-1, -1), 5); // aplicamos 5 dilataciones para tapar agujeros
	erode(dilata, clasificar_morf, kernel, Point(-1, -1), 5); // aplicamos 5 erociones para recuperar el tamaño original

// encontrar los contornoos de la imagen a clasificar
		
	findContours(clasificar_morf, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

	cout << endl;
	cout << "---------------------------------------------------" << endl;
	cout << "Clasificacion" << endl;
	cout << "El numero total de piezas" << endl;
	cout << " a clasificar es: " << contours.size() << endl;

	Mat contour_clasiff(clasificar_morf.size(), CV_8UC3, Scalar(255, 255, 255));

// se muestran los contornos hallados	
	Scalar colors1 = Scalar(0, 0, 0);
	
	for (int idx = 0; idx < contours.size(); idx++)
		drawContours(contour_clasiff, contours, idx, colors1);
	
	    namedWindow("Contornos imagen de clasificacion", CV_WINDOW_AUTOSIZE);
	    imshow("Contornos imagen de clasificacion", contour_clasiff);
	    cvWaitKey(0);

//////////////////////////////////////////////////////////////////////////////////////////////////

// CLASIFICACION
		// a) creando una matriz de datos de todos los contorno y efectuar la prediccion
		// b) efectuando la prediccion contorno por contorno

/*
// ///////////     CLASIFICACION a) prediccion con todas las muestras juntas !!!!!!!!!!!!!!!!!!!!!!!!

// Creamos Matriz vacia para vectores de datos

	int num_clasif = contours.size();
	Mat clasif_data(num_clasif, 2, CV_32FC1);

// A partir de los contornos llenamos clasif_data con calculos de perimetro y area

	int j = 0;

	for (size_t idx = 0; idx < contours.size(); idx++)
	{
		area = contourArea(contours[idx], false);
		perimetro = arcLength(contours[idx], 1);

		clasif_data.at<float>(j, 0) = perimetro;
		clasif_data.at<float>(j, 1) = area;
		j++;
	}
// se completo la matriz clasif_data
// se muestran los datos resultantes de los objetos a clasificar

	cout << "El contenido de la matriz [clasif_data] es:" << endl;
	cout << clasif_data << endl;
	
// Se crea una matriz para alojar el resultado de la clasificacion

		 Mat resultados; // el formato lo da el metodo predict()
		
		 modelo_bayes->predict(clasif_data, resultados); 

		  Mat R = resultados.t(); // R es la traspuesta de resultados 
// En la traspuesta R se puden observar los resultados en una fila !!!!!

		 cout  << "La clasificacion resulto: " << endl;  
		 //cout << resultados << endl; // resulta una columna de muchas filas es mejor mostrar la traspuesta
		 cout  << endl;
		 cout << "La clase de los objetos es:" << endl;
		 cout << R << endl; // es una fila de muchas columnas
		
		 system("PAUSE");
		 exit(1);
 
/*/
// /////////////  CLASIFICACION b) prediccion muestra a muestra !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

//*
// Se crea una una matriz para alojar solo un perimetro y un area
// Se crea una matriz para alojar solo un resultado

			  Mat un_objeto (1, 2, CV_32FC1);
	          Mat resultado (1, 1, CV_32FC1);
		      
// A partir de cada contorno completamos la matriz un_objeto y efectuamos la prediccion	
// y simultaneamente resaltamos el contorno en la imagen original
			  
			  for (size_t idx = 0; idx < contours.size(); idx++)
		      {
			  area = contourArea(contours[idx], false);
			  perimetro = arcLength(contours[idx], 1);
			  
			  un_objeto.at<float>(0, 0) = perimetro;
			  un_objeto.at<float>(0, 1) = area;
					 
		      resultado.at<float>(0,0) = modelo_bayes->predict(un_objeto); 
			
			  cout << endl;
			  cout << " el objeto resaltado " << " indice = " << idx <<endl;
			  cout << " es clase " << resultado << endl;
              cout << " con: perimetro,area" << un_objeto << endl;   
			  cout << " que corresponde a: " << nombre_clases[(int)resultado.at<float>(0, 0)] << endl;
			  cout << " -------------------------------------" << endl;
			  

			 cvtColor(clasificar, imagen_color, CV_GRAY2BGR);
			 Scalar color_rojo(0, 0, 255);
			 drawContours(imagen_color, contours, idx, color_rojo);
			 imshow("Clasificacion", imagen_color);
			 cvWaitKey(0);
			 }
//*/
		 
				return 0;

			
	}

		




		


		
	



	










