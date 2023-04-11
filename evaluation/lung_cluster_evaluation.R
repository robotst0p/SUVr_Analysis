rm(list=ls())

lungdata = read.csv(file = "/path/to/csv/file/with/survivor_data and clustering assignments")
names(lungdata) = c('TCGA_ID', 'HM90_K6_cluster', 'Stage', 'Gender', 'Vital_Status', 'Days_survival', 'Age', 'Residual_tumor','CC','NBS')
print(dim(lungdata))

library(survival)

sb = list()

max = lungdata$Days_survival[which.max(unlist(lungdata$Days_survival))]
min = lungdata$Days_survival[which.min(unlist(lungdata$Days_survival))]
delta = max - min 

for (i in 1:303)
{
  
  scaled_value = 1/delta * (lungdata$Days_survival[[i]] - max) + 1
  sb = append(sb,list(scaled_value))
}

print(length(lungdata$Vital_Status))
kmf <- survfit(Surv(as.numeric(sb), lungdata$Vital_Status=='deceased') ~ lungdata$NBS)
summary(kmf)
plot(kmf, mark=3, col = c(1,2,3,4,5,6), cex.lab=1.4,lwd=4,bty='n', xlab='Time', ylab='Survival probability', xaxs='i', yaxs='i', xlim=c(0,1), ylim=c(0,1))
legend("topright", legend=c("cluster 1","cluster 2", "cluster 3" , "cluster 4", "cluster 5", "cluster 6"), lwd=4,col=c(1,2,3,4,5,6), lty=1, cex=1)

survdiff(Surv(as.numeric(sb), lungdata$Vital_Status=='deceased') ~ lungdata$CC, rho=0)

lcox <- coxph(formula=Surv(as.numeric(sb), lungdata$Vital_Status=='deceased') ~ as.factor(lungdata$CC))
summary(lcox)

##############calc angles#################

library(xlsx)

rad2deg <- function(rad) {(rad * 180) / (pi)}
deg2rad <- function(deg) {(deg * pi) / (180)}

slope <- function(x,y) {
  (y-1)/(x-0)
}
fraction <- function(k1,k2){
  abs((k2-k1) / (1 + k2*k1))
}

create_points <- function(n, survfit_object){
  j = 1
  prev = 1
  time_1 = list()
  surv_1 = list()
  event_1=list()
  for ( i in n){
    time_1[j] = list(survfit_object$time[prev:(prev+i-1)])
    surv_1[j] = list(survfit_object$surv[prev:(prev+i-1)])
    event_1[j] = list(survfit_object$n.event[prev:(prev+i-1)])
    j = j + 1;
    prev = prev+i;
  }
  result <- c(list(time_1, surv_1, event_1))
}

calculate_angle<- function(points, curve_number){
  
  k1 = (0-1)/(1-0)
  theta=list()
  j = 1
  event = points[3][[1]][[curve_number]]
  survival = points[2][[1]][[curve_number]]
  time = points[1][[1]][[curve_number]]
  for (i in 1:length(head(event,-1))){
    if (event[i] == 1 ){
      if (time[i] > 0){
        k2= slope(time[i],survival[i])
        theta[j] <- rad2deg(atan(fraction(k1,k2)))
      } else {
        theta[j] <- 45
      }
      j = j+1  
    }
    
  }
  k2= slope(time[length(event)],survival[length(event)])
  theta[j] <- rad2deg(atan(fraction(k1,k2)))
  
  #theta
  angle = Reduce("+",theta)/length(theta) 
}


num_curves <-as.integer(kmf$strata)
points_1 = create_points(num_curves, kmf);

angle_1 = -1
angle_2 = -1
angle_3 = -1
angle_4 = -1
angle_5 = -1
angle_6 = -1

if ((!is.na(kmf$n[1])) & (kmf$n[1]> 1)){
  angle_1 = calculate_angle(points_1, 1)
}

if ((!is.na(kmf$n[2])) & (kmf$n[2]> 1)){
  angle_2 = calculate_angle(points_1, 2)
}

if ((!is.na(kmf$n[3])) & (kmf$n[3]> 1)){
  angle_3 = calculate_angle(points_1, 3)
}

if((!is.na(kmf$n[4])) & (kmf$n[4]> 1)){
  angle_4 = calculate_angle(points_1, 4)
}

if((!is.na(kmf$n[5])) & (kmf$n[5]> 1)){
  angle_5 = calculate_angle(points_1, 5)
}

if((!is.na(kmf$n[6])) & (kmf$n[6]> 1)){
  angle_6 = calculate_angle(points_1, 6)
}
list_points <- data.frame(x = 1:6, y = c(angle_1, angle_2, angle_3, angle_4, angle_5, angle_6), z = c(kmf$n[1],kmf$n[2],kmf$n[3],kmf$n[4],kmf$n[5], kmf$n[6]))
svd_weighted <- list_points$y[1] * list_points$z[1] / sum(list_points$z) + list_points$y[2] * list_points$z[2] / sum(list_points$z) + list_points$y[3] * list_points$z[3] / sum(list_points$z) + list_points$y[4] * list_points$z[4] / sum(list_points$z) +list_points$y[5] * list_points$z[5] / sum(list_points$z) +list_points$y[6] * list_points$z[6] / sum(list_points$z)
