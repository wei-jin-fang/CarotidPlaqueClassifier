suppressPackageStartupMessages({library(survival);library(ggplot2);library(ragg)})
O<-commandArgs(TRUE)[1];z<-readRDS(file.path(O,'final_analysis.rds'));d<-z$d;P<-z$P
clinical<-c('age','male','smoking','hypertension','diabetes','coronary_disease','ldl_c');full<-c(clinical,'stenosis3','antiplatelet','lipid_lowering')
form<-function(v,response='Surv(time_months,event)')as.formula(paste(response,'~',paste(v,collapse='+')))
out<-function(x,f)write.csv(x,file.path(O,f),row.names=FALSE)
out(data.frame(predictor=c('stenosis','echo','homogeneity','surface','ulcer','calcification'),reference=sapply(c('stenosis','echo','homogeneity','surface','ulcer','calcification'),function(k)levels(d[[k]])[1])),'US_reference_categories.csv')
extract<-function(f,key,ex){s<-summary(f);j<-match(ex,rownames(s$coefficients));data.frame(model=key,exposure=ex,n=f$n,events=f$nevent,HR=s$conf.int[j,1],low=s$conf.int[j,3],high=s$conf.int[j,4],p=s$coefficients[j,'Pr(>|z|)'])}
alt<-list();interaction<-list();tv<-list();landmark<-list()
for(ex in c('ai10','ai_binary')){
 for(key in c('four_categories','stratified_baseline')){v<-if(key=='four_categories')c(clinical,'stenosis4','antiplatelet','lipid_lowering') else c(clinical,'strata(stenosis3)','antiplatelet','lipid_lowering');f<-coxph(form(c(ex,v)),d,ties='efron',robust=TRUE);alt[[length(alt)+1]]<-extract(f,key,ex)}
 base<-coxph(form(c(ex,full)),d,ties='efron');fi<-coxph(form(c(paste0(ex,'*stenosis3'),clinical,'antiplatelet','lipid_lowering')),d,ties='efron');lr<-2*diff(c(base$loglik[2],fi$loglik[2]));interaction[[ex]]<-data.frame(exposure=ex,LR=lr,df=2,p=pchisq(lr,2,lower.tail=FALSE))
 for(t in c(3,6,12)){q<-d[d$time_months>t,];q$time_months<-q$time_months-t;f<-coxph(form(c(ex,full)),q,ties='efron',robust=TRUE);r<-extract(f,'Landmark',ex);r$landmark_month=t;landmark[[length(landmark)+1]]<-r}
}
f<-coxph(form(c('ai10','tt(ai10)',clinical,'antiplatelet','lipid_lowering','strata(stenosis3)')),d,ties='efron',tt=function(x,t,...)x*log(pmax(t,.1)/12));v<-vcov(f);b<-coef(f)
for(t in c(6,12,18,24,30,36)){L<-setNames(rep(0,length(b)),names(b));L['ai10']<-1;L['tt(ai10)']<-log(t/12);a<-sum(L*b);se<-sqrt(drop(t(L)%*%v%*%L));tv[[as.character(t)]]<-data.frame(month=t,HR=exp(a),low=exp(a-1.96*se),high=exp(a+1.96*se),p=2*pnorm(-abs(a/se)))}
out(do.call(rbind,alt),'Alternative_Cox.csv');out(do.call(rbind,interaction),'Stenosis_interaction.csv');out(do.call(rbind,tv),'Smooth_time_varying.csv');out(do.call(rbind,landmark),'Landmark_sensitivity.csv')
sp<-survSplit(Surv(time_months,event)~.,d,cut=c(12,24),episode='period',start='start');periodph<-list();timeint<-list()
for(ex in c('ai10','ai_binary')){
 intervals<-if(ex=='ai_binary')2:3 else 1:3;q<-sp[sp$period%in%intervals,];terms<-paste0('AI_period',intervals)
 for(i in intervals)q[[paste0('AI_period',i)]]<-q[[ex]]*(q$period==i)
 f<-coxph(form(c(terms,full,'strata(period)'),response='Surv(start,time_months,event)'),q,ties='efron',x=TRUE)
 zz<-cox.zph(f,transform='km');periodph[[ex]]<-data.frame(model='M4 period-specific',exposure=ex,term=rownames(zz$table),zz$table,row.names=NULL)
 common<-coxph(form(c(ex,full,'strata(period)'),response='Surv(start,time_months,event)'),q,ties='efron');lr<-2*(f$loglik[2]-common$loglik[2]);df<-length(intervals)-1
 timeint[[ex]]<-data.frame(exposure=ex,LR=lr,df=df,p=pchisq(lr,df,lower.tail=FALSE))
}
out(do.call(rbind,periodph),'Piecewise_PH_tests.csv');out(do.call(rbind,timeint),'Time_interaction.csv')
# Greedy nearest-neighbor matching, exact on stenosis and both baseline treatments.
ps<-glm(reformulate(c(clinical,'stenosis3','antiplatelet','lipid_lowering'),'ai_binary'),d,family=binomial());d$pslogit<-predict(ps,type='link');caliper<-.2*sd(d$pslogit)
stratum<-interaction(d$stenosis3,d$antiplatelet,d$lipid_lowering,drop=TRUE);pairs<-list()
for(g in levels(stratum)){
 hi<-which(stratum==g&d$ai_binary==1);lo<-which(stratum==g&d$ai_binary==0);hi<-hi[order(d$pslogit[hi],decreasing=TRUE)]
 for(i in hi){if(!length(lo))break;dist<-abs(d$pslogit[lo]-d$pslogit[i]);j<-which.min(dist);if(dist[j]<=caliper){pairs[[length(pairs)+1]]<-c(i,lo[j]);lo<-lo[-j]}}
}
pa<-do.call(rbind,pairs);md<-rbind(d[pa[,1],],d[pa[,2],]);md$pair_id<-rep(seq_len(nrow(pa)),2);md$group<-ifelse(md$ai_binary==1,'Higher risk','Lower risk');out(md,'Matched_data_private.csv');out(data.frame(pair_id=seq_len(nrow(pa)),high=d$patient_id[pa[,1]],low=d$patient_id[pa[,2]]),'Matched_pairs_private.csv')
bal<-list();for(g in levels(d$stenosis3)){key<-paste0('stenosis_',g);d[[key]]<-as.integer(d$stenosis3==g);md[[key]]<-as.integer(md$stenosis3==g)}
for(v in c(clinical,'antiplatelet','lipid_lowering',paste0('stenosis_',levels(d$stenosis3)))){a<-d[d$ai_binary==1,v];b<-d[d$ai_binary==0,v];s<-sqrt((var(a)+var(b))/2);ah<-md[md$ai_binary==1,v];al<-md[md$ai_binary==0,v];bal[[v]]<-data.frame(variable=v,before_SMD=(mean(a)-mean(b))/s,after_SMD=(mean(ah)-mean(al))/s,matched_low=mean(al),matched_high=mean(ah))};bal<-do.call(rbind,bal);out(bal,'Matching_balance.csv')
risks<-function(q){s<-survfit(Surv(time_months,event)~1,q);v<-summary(s,times=c(12,24,36),extend=TRUE)$surv; ix<-which(s$time<36);tt<-c(0,s$time[ix],36);ss<-c(1,s$surv[ix]);c(1-v,RMST=sum(diff(tt)*ss))}
h<-md[md$ai_binary==1,];l<-md[md$ai_binary==0,];rh<-risks(h);rl<-risks(l);set.seed(20261104);boot<-matrix(NA_real_,2000,4)
for(i in 1:2000){ix<-sample.int(nrow(h),replace=TRUE);boot[i,]<-risks(h[ix,])-risks(l[ix,])}
ci<-apply(boot,2,quantile,c(.025,.975));out(data.frame(horizon=c(12,24,36),high_risk=rh[1:3],low_risk=rl[1:3],difference=(rh-rl)[1:3],low=ci[1,1:3],high=ci[2,1:3],pairs=nrow(pa)),'Matched_risks.csv');out(data.frame(high_RMST=rh[4],low_RMST=rl[4],difference=rh[4]-rl[4],low=ci[1,4],high=ci[2,4]),'Matched_RMST.csv')
mc<-list();f<-coxph(Surv(time_months,event)~ai_binary+cluster(pair_id),md,ties='efron');mc[[1]]<-extract(f,'Pair-clustered full period','ai_binary');f<-coxph(Surv(time_months,event)~ai_binary+strata(pair_id),md,ties='efron');mc[[2]]<-extract(f,'Pair-stratified full period','ai_binary');out(do.call(rbind,mc),'Matched_Cox.csv')
out(data.frame(pairs=nrow(pa),caliper_logit=caliper,max_abs_SMD=max(abs(bal$after_SMD)),method='Greedy nearest-neighbor, descending propensity order; exact severity and baseline drugs; no replacement'),'Matching_summary.csv')
# Plot consistent panel sizes, solid model lines, legends to the right.
palette<-c(AI_probability='#3480b8',AI_classification='#82afda',US_continuous='#f79059',Clinical='#777777',Clinical_AI_probability='#8dcec8',Clinical_AI_classification='#c2bdde',Clinical_US='#c82423',Clinical_US_CT='#3480b8',Clinical_stenosis_treatment='#777777',Clinical_stenosis_treatment_CT='#3480b8')
labels<-c(AI_probability='CT probability',AI_classification='CT classification',US_continuous='US continuous',Clinical='Clinical',Clinical_AI_probability='Clinical + CT probability',Clinical_AI_classification='Clinical + CT classification',Clinical_US='Clinical + US continuous',Clinical_US_CT='Clinical + US + CT',Clinical_stenosis_treatment='Clinical + stenosis + treatment',Clinical_stenosis_treatment_CT='Base + CT')
# Aggregate equal score thresholds together; in particular binary ROC has one interior point.
rc<-list()
sfG<-survfit(Surv(time_months,as.integer(event==0 & time_months<36))~1,d)
getG<-function(t,left=FALSE)vapply(t,function(v){ix<-which(if(left)sfG$time<v else sfG$time<=v);if(length(ix))sfG$surv[max(ix)] else 1},0.)
for(horizon in c(12,24,36)){
 yy<-as.integer(d$event==1 & d$time_months<=horizon);ww<-rep(0,nrow(d));ww[yy==1]<-1/getG(d$time_months[yy==1],TRUE);ok<-yy==0 & d$time_months>=horizon;ww[ok]<-1/getG(rep(horizon,sum(ok)))
 for(key in names(P[[as.character(horizon)]])){
  pr<-P[[as.character(horizon)]][[key]];ord<-order(pr,decreasing=TRUE);ends<-cumsum(rle(pr[ord])$lengths)
  tpr<-cumsum(ww[ord]*yy[ord])/sum(ww*yy);fpr<-cumsum(ww[ord]*(1-yy[ord]))/sum(ww*(1-yy))
  rc[[length(rc)+1]]<-data.frame(horizon=horizon,model=key,FPR=c(0,fpr[ends]),TPR=c(0,tpr[ends]))
 }
}
roc<-do.call(rbind,rc);out(roc,'ROC.csv');cal<-read.csv(file.path(O,'Calibration.csv'));dc<-read.csv(file.path(O,'DCA.csv'));met<-z$metrics
png<-function(name,w,h,fun){agg_png(file.path(O,name),width=w,height=h,units='in',res=240);fun();dev.off()}
rp<-function(h,keys,title){par(mar=c(4,4,3,12),xpd=FALSE);plot(0,0,type='n',xlim=c(0,1),ylim=c(0,1),xlab='1 - Specificity',ylab='Sensitivity',main=title);abline(0,1,col='grey70',lty=2);for(k in keys){a<-roc[roc$horizon==h&roc$model==k,];lines(a$FPR,a$TPR,col=palette[k],lwd=2)};par(xpd=NA);legend(1.05,1,labels[keys],col=palette[keys],lty=1,lwd=2,bty='n',cex=.65);par(xpd=FALSE)}
png('Supplementary_Fig_11_updated.png',15,12,function(){par(mfrow=c(3,2));for(h in c(12,24,36)){rp(h,c('AI_probability','AI_classification','US_continuous','Clinical'),paste(h,'months'));rp(h,c('Clinical_AI_probability','Clinical_AI_classification','Clinical_US'),paste(h,'months'))}})
png('Supplementary_Fig_13_updated.png',16,5,function(){par(mfrow=c(1,3));for(h in c(12,24,36))rp(h,c('Clinical_stenosis_treatment','Clinical_stenosis_treatment_CT'),paste(h,'months'))})
png('Supplementary_Fig_12_updated.png',13,4.5,function(){par(mfrow=c(1,3),mar=c(4,4,3,1));for(h in c(12,24,36)){a<-cal[cal$horizon==h&cal$model=='AI_probability',];plot(a$predicted,a$observed,xlim=c(0,1),ylim=c(0,1),xlab='Frozen pathology-trained CT output',ylab='KM cumulative event risk',main=paste(h,'months'),pch=16,col='#3480b8');arrows(a$predicted,a$low,a$predicted,a$high,angle=90,code=3,length=.03,col='#3480b8');abline(0,1,col='grey60',lty=2)}})
png('Supplementary_Fig_14_updated.png',15,12,function(){par(mfrow=c(3,3));for(h in c(12,24,36)){keys<-c('Clinical_US','Clinical_US_CT');rp(h,keys,paste(h,'months: ROC'));par(mar=c(4,4,3,1));plot(0,0,type='n',xlim=c(0,.3),ylim=c(0,.3),xlab='OOF predicted event risk',ylab='KM cumulative event risk',main='Calibration');abline(0,1,col='grey60',lty=2);for(k in keys){a<-cal[cal$horizon==h&cal$model==k,];lines(a$predicted,a$observed,type='b',col=palette[k],pch=16)};plot(0,0,type='n',xlim=c(.01,.15),ylim=c(-.01,.1),xlab='Event-risk threshold',ylab='IPCW net benefit',main='Exploratory decision curve');abline(h=0,lty=2,col='grey60');a<-dc[dc$horizon==h&dc$model==keys[1],];lines(a$threshold,a$Treat_all,col='grey70');for(k in keys){a<-dc[dc$horizon==h&dc$model==k,];lines(a$threshold,a$NB,col=palette[k],lwd=2)}}})
png('Figure_7_updated.png',13,10,function(){par(mfrow=c(2,2),mar=c(4,4,3,1));for(keys in list(c('AI_probability','AI_classification','US_continuous','Clinical'),c('Clinical_AI_probability','Clinical_AI_classification','Clinical_US'))){plot(0,0,type='n',xlim=c(10,38),ylim=c(.5,1.02),xlab='Prediction horizon (months)',ylab='IPCW time-dependent AUC');for(k in keys){a<-met[met$model==k,];lines(a$horizon,a$AUC,type='b',pch=1,col=palette[k],lwd=2);arrows(a$horizon,a$AUC_low,a$horizon,a$AUC_high,col=palette[k],angle=90,code=3,length=.04)};legend('bottomleft',labels[keys],col=palette[keys],lty=1,pch=1,bty='n',cex=.7)};for(fam in c('CT_vs_US','Clinical_CT_vs_US')){a<-z$differences[z$differences$family==fam,];plot(a$delta_AUC,a$horizon,xlim=range(c(0,a$low,a$high)),ylim=c(8,40),xlab='Paired AUC difference',ylab='Prediction horizon (months)',main=fam,pch=1,col='#3480b8');segments(a$low,a$horizon,a$high,a$horizon,col='#3480b8',lwd=2);abline(v=0,col='grey60',lty=2)}})
cx<-read.csv(file.path(O,'Adjusted_Cox.csv'));sg<-read.csv(file.path(O,'Stenosis_subgroups.csv'))
png('Supplementary_Fig_15_updated.png',12,9,function(){par(mfrow=c(2,2),mar=c(4,9,3,1));for(ex in c('ai10','ai_binary'))for(a in list(cx[cx$exposure==ex,],sg[sg$exposure==ex,])){labs<-if('stenosis'%in%names(a))a$stenosis else a$model;plot(a$HR,seq_len(nrow(a)),xlim=range(c(1,a$low,a$high)),yaxt='n',xlab='Full-period HR (95% CI): exploratory',ylab='',main=ex,pch=16,col='#3480b8');axis(2,at=seq_len(nrow(a)),labels=labs,las=1);segments(a$low,seq_len(nrow(a)),a$high,seq_len(nrow(a)),col='#3480b8');abline(v=1,lty=2,col='grey60')}})
png('Supplementary_Fig_16_updated.png',15,5,function(){par(mfrow=c(1,3),mar=c(4,9,3,1));plot(abs(bal$before_SMD),seq_len(nrow(bal)),xlim=range(c(0,.1,abs(bal$before_SMD))),yaxt='n',xlab='Absolute standardized difference',ylab='',pch=1);points(abs(bal$after_SMD),seq_len(nrow(bal)),pch=16,col='#3480b8');axis(2,seq_len(nrow(bal)),bal$variable,las=1,cex.axis=.6);abline(v=.1,lty=2,col='grey60');par(mar=c(4,4,3,1));plot(survfit(Surv(time_months,event)~ai_binary,md),mark.time=TRUE,col=c('#3480b8','#c82423'),xlab='Months',ylab='Event-free survival',conf.int=FALSE,ylim=c(.85,1));a<-read.csv(file.path(O,'Matched_risks.csv'));plot(a$difference,a$horizon,xlim=range(c(0,a$low,a$high)),ylim=c(8,40),xlab='KM risk difference (higher minus lower)',ylab='Months',pch=16,col='#3480b8');segments(a$low,a$horizon,a$high,a$horizon,col='#3480b8');abline(v=0,lty=2,col='grey60')})
cat('SENSITIVITY AND FIGURES COMPLETE\n')
png('Figure_6_updated.png',13,8,function(){
 layout(matrix(c(1,2,3,4),2,2,byrow=TRUE),heights=c(4,1.3))
 for(ex in c('ai_binary','us_binary')){
  sf<-survfit(Surv(time_months,event)~d[[ex]],d);par(mar=c(2,4,3,1))
  plot(sf,mark.time=TRUE,col=c('#3480b8','#c82423'),lwd=2,conf.int=FALSE,xlim=c(0,36),ylim=c(.85,1),xaxt='n',xlab='',ylab='Event-free survival',main=if(ex=='ai_binary')'A  CT classification' else 'B  Ultrasound risk grouping')
  axis(1,at=seq(0,36,6))
  legend('bottomleft',c('Lower risk','Higher risk'),col=c('#3480b8','#c82423'),lwd=2,bty='n')
  s<-survdiff(Surv(time_months,event)~d[[ex]],d);mtext(paste('Log-rank P',format.pval(pchisq(s$chisq,1,lower.tail=FALSE),digits=3,eps=.001)),side=3,line=.1,cex=.8)
 }
 for(ex in c('ai_binary','us_binary')){
  par(mar=c(4,4,1,1));plot(0,0,type='n',xlim=c(0,36),ylim=c(0,3),xaxt='n',yaxt='n',bty='n',xlab='Months since baseline imaging',ylab='')
  axis(1,at=seq(0,36,6));mtext('Number at risk',side=3,line=0,adj=0)
  for(g in 0:1){sub<-d[d[[ex]]==g,];nr<-sapply(seq(0,36,6),function(t)sum(sub$time_months>=t));text(seq(0,36,6),rep(2-g,7),nr,col=c('#3480b8','#c82423')[g+1],cex=.85)}
 }
})
png('Overview_KM.png',4.2,3,function(){par(mar=c(3,3,1,1));plot(survfit(Surv(time_months,event)~ai_binary,d),col=c('#3480b8','#c82423'),lwd=2,mark.time=TRUE,conf.int=FALSE,ylim=c(.85,1),xlab='Months',ylab='Event-free survival');legend('bottomleft',c('Lower risk','Higher risk'),col=c('#3480b8','#c82423'),lty=1,lwd=2,bty='n',cex=.7)})
png('Figure_7_updated.png',13,10,function(){
 par(mfrow=c(2,2),mar=c(4,4,3,1))
 sets<-list(c('AI_probability','AI_classification','US_continuous','Clinical'),c('Clinical_AI_probability','Clinical_AI_classification','Clinical_US'))
 for(i in 1:2){
  keys<-sets[[i]];plot(0,0,type='n',xlim=c(10,38),ylim=c(.5,1.02),xaxt='n',xlab='Prediction horizon (months)',ylab='IPCW time-dependent AUC',main=c('A  Standalone assessments','B  Clinical combinations')[i]);axis(1,c(12,24,36))
  for(k in keys){a<-met[met$model==k,];lines(a$horizon,a$AUC,type='b',pch=1,col=palette[k],lwd=2);arrows(a$horizon,a$AUC_low,a$horizon,a$AUC_high,col=palette[k],angle=90,code=3,length=.04)}
  legend('bottomleft',labels[keys],col=palette[keys],lty=1,pch=1,bty='n',cex=.75)
 }
 for(i in 1:2){
  fam<-c('CT_vs_US','Clinical_CT_vs_US')[i];a<-z$differences[z$differences$family==fam,]
  plot(a$delta_AUC,a$horizon,xlim=range(c(0,a$low,a$high)),ylim=c(8,40),yaxt='n',xlab='Paired AUC difference (95% CI)',ylab='Prediction horizon (months)',main=c('C  CT probability minus ultrasound','D  Clinical + CT minus clinical + US')[i],pch=1,col='#3480b8');axis(2,c(12,24,36));segments(a$low,a$horizon,a$high,a$horizon,col='#3480b8',lwd=2);abline(v=0,col='grey60',lty=2)
 }
})
