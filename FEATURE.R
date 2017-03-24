rm(list=ls())
library(MASS)
setwd('C:\\Users\\yangy_000\\Dropbox\\BAYLOR\\TEMP\\NeuralSignalDecoding')
for(j in 0:69)
{
  fname <- paste0("NODE",j,".csv")
  trial <- read.csv(fname, header = TRUE)
  feature <- matrix(0, nrow = length(levels(trial$phone.name)), ncol = 6)
  row.names(feature) <- levels(trial$phone.name)
  colnames(feature) <- c('Delta','Theta','Alpha','Beta','LowGamma','HighGamma')
  for(i in 1:6)
  {
    feature[,i] <- tapply(trial[,i + 2],trial$phone.name,mean)
  }
  rfname <- paste0('C:\\Users\\yangy_000\\Dropbox\\BAYLOR\\TEMP\\NeuralSignalDecoding\\FEATURE_',fname)
  write.csv(feature,rfname)
}
#############
rm(list=ls())
Nodename <- read.table('data\\electrode_names.txt')
#Set parameters of graphs
#par(mai=c(1, 1, 0.5, 0.5),
#   mgp=c(0, 0.5, 0),
#  tck=0)
# layout(mat=matrix(c(1,2,3,4),2,byrow=TRUE),
#        widths=c(1,1),
#        heights=c(1,1),
#        respect=FALSE)
############
library(gridGraphics)
library(grid)
library(gplots)
library(gridExtra)
grab_grob <- function(){
  grid.echo()
  grid.grab()
}

pdf(file= "PLOTS.pdf", onefile = TRUE)  
for(j in 0:1)#the (j + 1)th node
{
  fname <- paste0("FEATURE_NODE",j,".csv")
  
  feature <- read.csv(fname,header = TRUE)
  
  rownames(feature) <- feature$X
  feature <- feature[,-1]
  #Heatmap
  heatmap(cor(t(feature)), symm=T, main =  paste("Heatmap", Nodename[j + 1,1]))
  # heatmap.2(cor(t(feature)), dendrogram ='row',
  #           Colv=FALSE, col=greenred(800), 
  #           key=FALSE, keysize=1.0, symkey=FALSE, density.info='none',
  #           trace='none', colsep=1:10,
  #           sepcolor='white', sepwidth=0.05,
  #           scale="none",cexRow=0.2,cexCol=2,
  #           labCol = colnames(cor(t(feature))),                 
  #           hclustfun=function(c){hclust(c, method='mcquitty')},
  #           lmat=rbind( c(0, 3), c(2,1), c(0,4) ), lhei=c(0.25, 4, 0.25 ),                 
  # )
  g1 <- grab_grob()
  #Barplot
  mp <- barplot(feature[,6], las=2, main =  Nodename[j + 1,1] )
  text(mp, par("usr")[3], labels = rownames(feature), 
       srt = 90, adj = c(1.1,1.1), xpd = TRUE, cex=1)
  g2 <- grab_grob()
  #Classic MDS
  fitMDS <- cmdscale(1-cor(t(feature)), k = 2, eig = TRUE)
  
  plot(cmdscale(1-cor(t(feature))), xlab = "",ylab = "", type = 'n',
       main = paste("Classic MDS",Nodename[j + 1,1]))
  text(cmdscale(1-cor(t(feature))), labels = rownames(feature))
  #colscl = colorRampPalette(c('navy', 'white', 'orangered'))(100)
  g3 <- grab_grob()
  #Nonmetric MDS
  fit <- isoMDS(1-cor(t(feature)),k = 2)
  fit$stress
  plot(fit$points[,1],fit$points[,2],type = 'n',
       main = paste("Nonmetric MDS",Nodename[j + 1,1]),
       xlab = "" , ylab = "")
  text(fit$points[,1],fit$points[,2], labels = rownames(feature))
  g4 <- grab_grob()
  grid.newpage()
  grid.arrange(g1,g2,g3,g4, ncol=2, clip=TRUE)
}

dev.off() 


#image(t(cor(t(feature), method='pearson')), x=1:41, y=1:41, col=colscl, zlim=c(-1,1))





