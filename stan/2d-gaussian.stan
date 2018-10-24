functions {
    real yield(real y_norm,
               real[] d_t, 
               //real d_p, 
               real mu_t, 
               //real mu_p, 
               real sigma_t, 
               //real sigma_p, 
               //real rho
               ){
        real dy[6];
        for (m in 1:6){
            dy[m] = y_norm * (exp(-0.5*square((d_t[m]-mu_t)/sigma_t)));
        }
        return sum(dy);
        // return d_t; //test function can run 
        //return (1/(2 * pi() * sigma_t * sigma_p * sqrt(1 - square(rho)))) * 
        //    exp( 
        //        - (1/(2* (1- square(rho) ))) * 
        //        (
        //            square((d_t - mu_t)/(sigma_t))
        //            + square((d_p - mu_p)/(sigma_p))
        //            - (2* rho * (d_t - mu_t) * (d_p - mu_p) / (sigma_p * sigma_p) ) 
        //        )
        //    );
 
    }
}

data {
    int<lower=0> n_regions;
    int<lower=0> n_years;
    real d_temp[n_regions,n_years,6];
    //real d_precip[n_regions,n_years,6;
    real d_yields[n_regions,n_years];
    int n_gf;
    real temp[n_gf];
}

parameters {
    real mu_t;
    //real mu_p;
    //Wikipedia notation
    real<lower=0.0> y_norm;
    real<lower=0.0> sigma_t;
    //real sigma_p;
    //real rho;
    //real noise_sigma[n_regions];
}


model {

    real d_y;
    
    //Priors
    mu_t      ~normal(20.0,5.0);
    //mu_p      ~normal(0.0,50.0);
    y_norm    ~normal(1.0,3.0);
    sigma_t   ~normal(5.0,1.0);
    //sigma_p   ~normal(50.0,25.0);
    //rho       ~uniform(-0.9,0.9);

    //for (m in 1:12){
    //    y_norm[n_regions,m]      ~normal(1.0,3.0);
    //    //noise_sigma[n_regions] ~normal(0.0,100.0);
    //    }

    for (n in 1:n_regions){
        
        for (y in 1:n_years){
            d_y = 0.0;
            //for (m in 1:12){
                d_y =  yield(y_norm,
                               d_temp[n,y,:], 
                               //d_precip[n,y,:], 
                               mu_t, 
                               //mu_p, 
                               sigma_t, 
                               //sigma_p, 
                               //rho
                               );

            //}
            d_yields[n,y] ~normal(d_y, 1.0);
        }
    }
}


//generated quantities {
//    real d_yields_pred[n_regions,n_years];
//    real tmp;
//    for (n in 1:n_regions){
//        for (y in 1:n_years){
//            tmp=0.0;
//            for (m in 1:12){
//                tmp=tmp+s_temp[n,m]*d_temp[n,y,m] + s_precip[n,m]*d_precip[n,y,m];
//            }
//            d_yields_pred[n,y]=normal_rng(tmp,noise_sigma);
//        }
//    }
//}