rm(list=ls())
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
      k2= slope(time[i],survival[i])
      theta[j] <- rad2deg(atan(fraction(k1,k2)))
      j = j+1  
    }
    
  }
  k2= slope(time[length(event)],survival[length(event)])
  theta[j] <- rad2deg(atan(fraction(k1,k2)))
  
  #theta
  angle = Reduce("+",theta)/length(theta) 
}


path_to_file <- "path/to/folder/with/files/containing/assigned/clusters"
path_to_img <- "path/to/folder/to/store/images"
path_to_rdata <- "path/to/folder/to/store/r/data"
files <- list.files(path = path_to_file, pattern = "\\.csv$")
counter <- 0

for (f in files[sort.list(files)])
{
  counter <- counter + 1
  file_to_load = paste(path_to_file, f, sep = "/")
  lungdata = read.csv(file = file_to_load)
  # Last column contains Catalysis Clustering assignments
  names(lungdata) = c('TCGA_ID', 'HM90_K4_cluster_x', 'Stage', 'Gender', 'Vital_Status', 'Days_survival', 'Age', 'Residual_tumor', 'HM90_K4_cluster_y', 'HM90_K4_cluster')
  print(dim(lungdata))
  
  library(survival)
  
  sb = list()
  sb_1 = list()
  sb_2 = list()
  sb_3 = list()
  sb_4 = list()
  
  max = lungdata$Days_survival[which.max(unlist(lungdata$Days_survival))]
  min = lungdata$Days_survival[which.min(unlist(lungdata$Days_survival))]
  delta = max - min 
  
  for (i in 1:325) ##328
  {
    
    scaled_value = 1/delta * (lungdata$Days_survival[[i]] - max) + 1
    sb = append(sb,list(scaled_value))
  }
  
  print(length(lungdata$Vital_Status))
  kmf <- survfit(Surv(as.numeric(sb), lungdata$Vital_Status=='deceased') ~ lungdata$HM90_K4_cluster)
  summary(kmf)
  png(filename = paste(path_to_img, counter,'.png', sep = ""))
  plot(kmf, mark=3, col = c(1,2,3,4),cex.lab=1.4,lwd=4, bty='n', xlab='Time', ylab='Survival probability', xaxs='i', yaxs='i', xlim=c(0,1), ylim=c(0,1))
  legend("topright", lwd=4, legend=c("cluster 1","cluster 2", "cluster 3" , "cluster 4"), col=c(1,2,3,4), lty=1, cex=1)
  dev.off()
  survdiff(Surv(as.numeric(sb), lungdata$Vital_Status=='deceased') ~ lungdata$HM90_K4_cluster, rho=0)
  
  lcox <- coxph(formula=Surv(as.numeric(sb), lungdata$Vital_Status=='deceased') ~ as.factor(lungdata$HM90_K4_cluster))
  summary(lcox)
  
  num_curves <-as.integer(kmf$strata)
  points_1 = create_points(num_curves, kmf);
  
  angle_1 = -1
  angle_2 = -1
  angle_3 = -1
  angle_4 = -1
  
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
  
  list_points <- data.frame(x = 1:4, y = c(angle_1, angle_2, angle_3, angle_4), z = c(kmf$n[1],kmf$n[2],kmf$n[3],kmf$n[4]))
  total = sum(list_points$z,NA, na.rm = TRUE)
  Angle_new = 0
  for (k in 1:4){
    if (list_points$y[k] > 0){
      Angle_new = Angle_new + list_points$y[k] * list_points$z[k]
    }
  }
  Angle_new = Angle_new / total
  
  save(list_points, file = paste(path_to_rdata, counter,'.Rdata', sep = ""))
}


