suppressPackageStartupMessages({library(survival);library(ragg)})
O<-commandArgs(TRUE)[1];z<-readRDS(file.path(O,'final_analysis.rds'));d<-z$d;d$us_probability<-z$P[['36']]$US_continuous;d$us_binary<-as.integer(d$us_probability>=median(d$us_probability));cols<-c('#3480b8','#c82423')
stepxy<-function(t,y)cbind(x=c(0,rep(t,each=2)),y=c(1,as.vector(rbind(c(1,head(y,-1)),y))))
kmplot<-function(dat,ex,title,small=FALSE){
 plot(0,0,type='n',xlim=c(0,36),ylim=c(.85,1.005),xaxt='n',xlab=if(small)'Months' else '',ylab='Event-free survival',main=title)
 axis(1,at=seq(0,36,6))
 for(g in 0:1){
  sub<-dat[dat[[ex]]==g,];f<-survfit(Surv(time_months,event)~1,sub,conf.type='log-log')
  lo<-stepxy(f$time,f$lower);hi<-stepxy(f$time,f$upper)
  polygon(c(lo[,1],rev(hi[,1])),c(lo[,2],rev(hi[,2])),col=adjustcolor(cols[g+1],alpha.f=.18),border=NA)
 }
 for(g in 0:1){
  f<-survfit(Surv(time_months,event)~1,dat[dat[[ex]]==g,],conf.type='log-log');s<-stepxy(f$time,f$surv);lines(s[,1],s[,2],col=cols[g+1],lwd=if(small)1.6 else 2)
  ix<-f$n.censor>0;points(f$time[ix],f$surv[ix],pch=3,col=cols[g+1],cex=if(small).45 else .7)
 }
 legend('bottomleft',c('Lower risk','Higher risk'),col=cols,lwd=2,bty='n',cex=if(small).7 else 1)
}
agg_png(file.path(O,'Figure_6_updated.png'),width=13,height=8,units='in',res=240)
layout(matrix(c(1,2,3,4),2,2,byrow=TRUE),heights=c(4,1.3))
for(ex in c('ai_binary','us_binary')){par(mar=c(2,4,3,1));kmplot(d,ex,if(ex=='ai_binary')'A  CT classification' else 'B  Ultrasound risk grouping');s<-survdiff(Surv(time_months,event)~d[[ex]],d);mtext(paste('Log-rank P',format.pval(pchisq(s$chisq,1,lower.tail=FALSE),digits=3,eps=.001)),side=3,line=.1,cex=.8)}
for(ex in c('ai_binary','us_binary')){par(mar=c(4,4,1,1));plot(0,0,type='n',xlim=c(0,36),ylim=c(0,3),xaxt='n',yaxt='n',bty='n',xlab='Months since baseline imaging',ylab='');axis(1,at=seq(0,36,6));mtext('Number at risk',side=3,line=0,adj=0);for(g in 0:1){sub<-d[d[[ex]]==g,];text(seq(0,36,6),rep(2-g,7),sapply(seq(0,36,6),function(t)sum(sub$time_months>=t)),col=cols[g+1],cex=.85)}}
dev.off()
agg_png(file.path(O,'Overview_KM.png'),width=4.2,height=3,units='in',res=240);par(mar=c(4,4,1,1));kmplot(d,'ai_binary','',TRUE);dev.off()
md<-read.csv(file.path(O,'Matched_data_private.csv'));bal<-read.csv(file.path(O,'Matching_balance.csv'));a<-read.csv(file.path(O,'Matched_risks.csv'))
agg_png(file.path(O,'Supplementary_Fig_16_updated.png'),width=15,height=5,units='in',res=240)
par(mfrow=c(1,3),mar=c(4,9,3,1));plot(abs(bal$before_SMD),seq_len(nrow(bal)),xlim=range(c(0,.1,abs(bal$before_SMD))),yaxt='n',xlab='Absolute standardized difference',ylab='',pch=1,main='A  Matching balance');points(abs(bal$after_SMD),seq_len(nrow(bal)),pch=16,col=cols[1]);axis(2,seq_len(nrow(bal)),bal$variable,las=1,cex.axis=.6);abline(v=.1,lty=2,col='grey60');legend('bottomright',c('Before','After'),pch=c(1,16),col=c('black',cols[1]),bty='n',cex=.7)
par(mar=c(4,4,3,1));kmplot(md,'ai_binary','B  Matched survival',TRUE)
plot(a$difference,a$horizon,xlim=range(c(0,a$low,a$high)),ylim=c(8,40),xlab='KM risk difference (higher minus lower)',ylab='Months',pch=16,col=cols[1],main='C  Matched risk difference');segments(a$low,a$horizon,a$high,a$horizon,col=cols[1]);abline(v=0,lty=2,col='grey60');dev.off()
# Align the horizon-risk confidence limits with the plotted log-log method.
r<-read.csv(file.path(O,'KM_horizon_risks.csv'))
for(i in seq_len(nrow(r))){ex<-r$assessment[i];g<-if(r$group[i]=='Lower risk')0 else 1;f<-survfit(Surv(time_months,event)~1,d[d[[ex]]==g,],conf.type='log-log');s<-summary(f,times=r$horizon[i],extend=TRUE);r$low[i]<-1-s$upper;r$high[i]<-1-s$lower}
write.csv(r,file.path(O,'KM_horizon_risks.csv'),row.names=FALSE)
cat('KM 95% log-log CI bands, censor marks, risk counts updated.\n')
