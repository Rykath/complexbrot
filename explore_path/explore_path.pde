double cenCr = 0.0;
double cenCi = 0.0;
double resC = 5.0/1000;

int iter = 20;

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
  Cr = (mouseX-width/2.0)*resC+cenCr;
  Ci = (mouseY-height/2.0)*resC+cenCi;
  Zr = 0;
  Zi = 0;
  
  for (int i=0;i<iter;i++){
  stroke(255,(float)i/iter*255,(float)i/iter*255);
  Zr2 = Zr*Zr - Zi*Zi + Cr;
  Zi2 = 2*Zr*Zi + Ci;
  lineCplx(Zr,Zi,Zr2,Zi2);
  Zr = Zr2;
  Zi = Zi2;
  }
  
  // escape - targets
  stroke(255,150,0);
  for (int x=0;x<width;x++){
  for (int y=0;y<height;y++){
    Cr = (x-width/2.0)*resC+cenCr;
    Ci = (y-height/2.0)*resC+cenCi;
    if (Cr*Cr+Ci*Ci <= 4.0){ // i 1 inside
      // i 2
      Zr = Cr*Cr - Ci*Ci + Cr;
      Zi = 2*Cr*Ci + Ci;
      Zr3 = Zr;
      Zi3 = Zi;
      if (Zr*Zr+Zi*Zi <= 4.0){
        // i 3
        Zr2 = Zr*Zr - Zi*Zi + Cr;
        Zi2 = 2*Zr*Zi + Ci;
        if (Zr2*Zr2 + Zi2*Zi2 <= 4.0){
          // i 4
          Zr = Zr2*Zr2 - Zi2*Zi2 + Cr;
          Zi = 2*Zr2*Zi2 + Ci;
          if (Zr*Zr+Zi*Zi > 4.0){
            pointCplx(Zr3,Zi3);
          }
        }
      }
    }
  }
  }
      
  
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