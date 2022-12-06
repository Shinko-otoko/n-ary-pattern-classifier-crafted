
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>




void myFunction2(float *proba,float *moyenne,int nb_pixels, int len_x_train,int nb_classes)
{

    int compteur=0;
    for (int j=0;j<len_x_train;j++)
    {
        for (int k=0;k<nb_classes;k++)
        {
            moyenne[j*nb_classes+k]=0;
        }
        for (int i=0;i<nb_pixels;i++)
        {
            for (int k=0;k<nb_classes;k++)
            {
                moyenne[j*nb_classes+k]+=proba[compteur];
                compteur+=1;
            }
        }
    }

}



void myFunction(int *angles, int len_angles, int *distances, int len_distances, int nb_pixels ,int *b, int *g, int *r,int *l_x_c, int *l_y_c, int width, int height,int *result)
{
    //printf ("%d %d %d %d", nb_pixels, len_angles,width, height);
    // int **tab_pixels=(int **)malloc(nb_pixels*sizeof(int*)); 

    // for (int i=0;i<nb_pixels;i++)
    // {
    //     tab_pixels[i]=(int *)malloc ((len_angles+1)*3*sizeof(int));
    //     for (int j=0;j<(len_angles+1)*3;j++)
    //     {
    //         tab_pixels[i][j]=-1;
    //     }
    
    // }



    

    
    for (int i=0; i<nb_pixels;i++)
    {
        int compteur=3;
        int x= l_x_c[i];
        int y=l_y_c[i];

        result[i*(len_angles*3)]=b[x*width+y];
        result[i*(len_angles*3)+1]=g[x*width+y];
        result[i*(len_angles*3)+2]=r[x*width+y];

        for(int k=0;k<len_angles;k++)
        {
            int x_=(int)(x - distances[k]*sin(angles[k]));
            int y_=(int)(y + distances[k]*cos(angles[k]));
            if ((x_> 0 && x_<width) && (y_> 0 && y_<height))
            {
                result[i*(len_angles*3)+compteur]=b[x_*width+y_];
                result[i*(len_angles*3)+compteur+1]=g[x_*width+y_];
                result[i*(len_angles*3)+compteur+2]=r[x_*width+y_];
            }
            else 
            {
                 result[i*(len_angles*3)+compteur]=-1;
                result[i*(len_angles*3)+compteur+1]=-1;
                result[i*(len_angles*3)+compteur+2]=-1;               
            }

            compteur+=3;
        }

        
        //


    }


    
    
  


}
