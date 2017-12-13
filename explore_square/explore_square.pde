double cenCr = 0.0;
double cenCi = 0.0;
double resC = 5.0/1000;

int iter = 2;
int square = 8; // offset of square from mouse pointer in pixels

PImage bgI;
double Cr, Ci, Zr, Zi, Zr2, Zi2, Zr3, Zi3;

void setup(){
  size(1000,1000);
  bgI = loadImage("cu_m_1ksq_5w_1ki_-29s.png");
}

void draw(){
  // Background
  image(bgI,0,0);
  
  // mouse-path
  stroke(255,0,0);
  drawPath(mouseX,mouseY);
  
  // mouse-path polar
  stroke(0,255,0);
  
  Cr = (mouseX-width/2.0)*resC+cenCr;
  Ci = (mouseY-height/2.0)*resC+cenCi;
  
  float Cl = sqrt((float)(Cr*Cr + Ci*Ci));
  float Ca = atan2((float)Ci,(float)Cr);
  
  float Zl = 0;
  float Za = 0;
  float Zl2, Za2;
  
  for (int i=0;i<iter;i++){
    Zl2 = sqrt(pow(Zl,4) + pow(Cl,2) + 2*Cl*pow(Zl,2)*cos(2*Za-Ca));
    Za2 = 2*Za + atan2(Cl*sin(Ca-2*Za),Zl*Zl+Cl*cos(Ca-2*Za));
    lineCplx(Zl*cos(Za),Zl*sin(Za),Zl2*cos(Za2),Zl2*sin(Za2));
    Zl = Zl2;
    Za = Za2;
  }
  
  stroke(0,0,255);
  fill(0,0,255);
  Cr = (mouseX-width/2.0)*resC+cenCr;
  Ci = (mouseY-height/2.0)*resC+cenCi;
  Cl = sqrt((float)(Cr*Cr + Ci*Ci));
  Ca = atan2((float)Ci,(float)Cr);
  
  circCplx(0,0,5);
  circCplx(Cl*cos(Ca),Cl*sin(Ca),5);
  circCplx(Cl*Cl*cos(2*Ca)+Cl*cos(Ca),Cl*Cl*sin(2*Ca)+Cl*sin(Ca),5);
  
  // ## draw path for square
  //stroke(255,255,0);
  //drawPath(mouseX+square,mouseY+square);
  //stroke(255,0,255);
  //drawPath(mouseX+square,mouseY-square);
  //stroke(0,255,255);
  //drawPath(mouseX-square,mouseY+square);
  //stroke(255,100,100);
  //drawPath(mouseX-square,mouseY-square);
  
  // ## draw square for each iter
  //double[] aCr = new double[4];
  //double[] aCi = new double[4];
  //aCr[0] = (mouseX-width/2.0+square)*resC+cenCr;
  //aCi[0] = (mouseY-height/2.0+square)*resC+cenCi;
  //aCr[1] = (mouseX-width/2.0+square)*resC+cenCr;
  //aCi[1] = (mouseY-width/2.0-square)*resC+cenCi;
  //aCr[2] = (mouseX-width/2.0-square)*resC+cenCr;
  //aCi[2] = (mouseY-width/2.0-square)*resC+cenCi;
  //aCr[3] = (mouseX-width/2.0-square)*resC+cenCr;
  //aCi[3] = (mouseY-width/2.0+square)*resC+cenCi;
  //double[] aZr = new double[4];
  //double[] aZi = new double[4];
  //for (int i=0;i<4;i++){
  //  aZr[i] = 0;
  //  aZi[i] = 0;
  //}
  //for (int i=0;i<iter;i++){
  //  for (int p=0;p<4;p++){
  //    Zr2 = aZr[p]*aZr[p] - aZi[p]*aZi[p] + aCr[p];
  //    Zi2 = 2*aZr[p]*aZi[p] + aCi[p];
  //    aZr[p] = Zr2;
  //    aZi[p] = Zi2;
  //  }
  //  stroke(255,0,0);
  //  lineCplx(aZr[0],aZi[0],aZr[1],aZi[1]);
  //  stroke(255,50,50);
  //  lineCplx(aZr[1],aZi[1],aZr[2],aZi[2]);
  //  stroke(255,100,100);
  //  lineCplx(aZr[2],aZi[2],aZr[3],aZi[3]);
  //  stroke(255,150,150);
  //  lineCplx(aZr[3],aZi[3],aZr[0],aZi[0]);
  //}
  
  // escape - targets
  //stroke(255,150,0);
  //for (int x=0;x<width;x++){
  //for (int y=0;y<height;y++){
  //  Cr = (x-width/2.0)*resC+cenCr;
  //  Ci = (y-height/2.0)*resC+cenCi;
  //  if (Cr*Cr+Ci*Ci <= 4.0){ // i 1 inside
  //    // i 2
  //    Zr = Cr*Cr - Ci*Ci + Cr;
  //    Zi = 2*Cr*Ci + Ci;
  //    Zr3 = Zr;
  //    Zi3 = Zi;
  //    if (Zr*Zr+Zi*Zi <= 4.0){
  //      // i 3
  //      Zr2 = Zr*Zr - Zi*Zi + Cr;
  //      Zi2 = 2*Zr*Zi + Ci;
  //      if (Zr2*Zr2 + Zi2*Zi2 <= 4.0){
  //        // i 4
  //        Zr = Zr2*Zr2 - Zi2*Zi2 + Cr;
  //        Zi = 2*Zr2*Zi2 + Ci;
  //        if (Zr*Zr+Zi*Zi > 4.0){
  //          pointCplx(Zr3,Zi3);
  //        }
  //      }
  //    }
  //  }
  //}
  //}
  
  // grid
  stroke(0,0,255);
  line(width/2,0,width/2,height);
  line(0,height/2,width,height/2);
  stroke(0,0,200);
  noFill();
  ellipse(width/2,height/2,(int)(4/resC),(int)(4/resC)); // r=2
  ellipse(width/2,height/2,(int)(2/resC),(int)(2/resC)); // r=1
  ellipse(width/2,height/2,(int)(1/resC),(int)(1/resC)); // r=0.5
  
  // period 3 - solutions / centers
  stroke(0,255,0);
  fill(0,255,0);
  circCplx(0,0,5);
  circCplx(-1.7549,0,5);
  circCplx(-0.12256,0.74486,5);
  circCplx(-0.12256,-0.74486,5);
  
  // escape 2 region - maxima
  circCplx(-0.5,1.3229,5);
  circCplx(-0.5,-1.3229,5);
  noFill();
  ellipseCplx(-0.5,0,3,2*1.3229);
  
  // escape 3/4 boundary
  fill(0,255,0);
  // escaping at 2
  circCplx(-2,0,5);
  circCplx(0.68233,0,5);
  circCplx(-0.3411,1.16154,5);
  circCplx(-0.3411,-1.16154,5);
  // escaping at -2
  circCplx(0.37528,-0.89540,5);
  circCplx(-1.3753,-0.4801,5);
  // escaping at +- 2i
  circCplx(-1.8681,-0.2673,5);
  circCplx(-0.78232,-0.92581,5);
  circCplx(0.05435,-1.13996,5);
  circCplx(0.59602,-0.48143,5);
  
  delay(50);
}

void lineCplx(double ar,double ai,double br,double bi){
  line((int)((ar+cenCr)/resC+width/2.0),(int)((ai+cenCi)/resC+height/2.0),(int)((br+cenCr)/resC+width/2.0),(int)((bi+cenCi)/resC+height/2.0));
}

void pointCplx(double xr, double xi){
  point((int)((xr+cenCr)/resC+width/2.0),(int)((xi+cenCi)/resC+height/2.0));
}

void circCplx(double xr, double xi, int r){
  ellipse((int)((xr+cenCr)/resC+width/2.0),(int)((xi+cenCi)/resC+height/2.0),r,r);
}

void ellipseCplx(double xr, double xi, double w, double h){
  ellipse((int)((xr+cenCr)/resC+width/2.0),(int)((xi+cenCi)/resC+height/2.0),(int)(w/resC),(int)(h/resC));
}

void drawPath(int x, int y){
  Cr = (x-width/2.0)*resC+cenCr;
  Ci = (y-height/2.0)*resC+cenCi;
  Zr = 0;
  Zi = 0;
  for (int i=0;i<iter;i++){
    Zr2 = Zr*Zr - Zi*Zi + Cr;
    Zi2 = 2*Zr*Zi + Ci;
    lineCplx(Zr,Zi,Zr2,Zi2);
    Zr = Zr2;
    Zi = Zi2;
  }
}