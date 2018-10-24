functions {
    real yield(real[] temp,  
               real mu_t,   
               real sigma_t,  
               //real[] precip,  
               //real mu_p,   
               //real sigma_p, 
               real norm){
        real dy[6];
        int reci;
        for (i in 1:6){
            reci = i;
            dy[reci]=norm*exp(-0.5*( square( (temp[reci]-mu_t)/sigma_t)  ) );
            //dy[i]=norm*exp(-0.5*( square( (temp[i]-mu_t)/sigma_t) +  square( (precip[i]- mu_p)/sigma_p) ) );
        }
        return sum(dy);
    }
}

data {
    int<lower=0> n_regions;
    int<lower=0> n_years;
    real d_temp[n_regions,n_years,6];
    real d_precip[n_regions,n_years,6];
    real d_yields[n_regions,n_years];
    int n_gf;
    real temp[n_gf];
}



parameters {
    real mu_t;
    real<lower=0.0> sigma_t;
    real mu_p;
    real<lower=0.0> sigma_p;
    real<lower=0.0> norm;
}

model {

    mu_t ~ normal(20,5);
    sigma_t ~ normal(5,1);
    mu_p ~ normal(20,5);
    sigma_p ~ normal(5,1);
    norm ~ normal(1,3);
    for (n in 1:n_regions){
        for (y in 1:n_years){       
            d_yields[n,y]~normal(yield(temp,  
                                       mu_t,   
                                       sigma_t,  
                                       //precip,  
                                       //mu_p,  
                                       //sigma_p,  
                                       norm),
                                  1.0);
        }
    }
}


generated quantities {
real fdy[n_gf];
real pred_yields[n_regions,n_years];
for (i in 1:n_gf){
fdy[i]=norm*exp(-0.5*square((temp[i]-mu_t)/sigma_t));
 }
for (n in 1:n_regions){
for (y in 1:n_years){
pred_yields[n,y]=normal_rng(yield(d_temp[n,y,:],mu_t, sigma_t, norm),1.0);
}
}}