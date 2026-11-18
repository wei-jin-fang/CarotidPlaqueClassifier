suppressPackageStartupMessages({library(survival);library(glmnet);library(ggplot2);library(ragg)})
args<-commandArgs(TRUE);O<-args[1];dir.create(O,recursive=TRUE,showWarnings=FALSE)
d<-read.csv(file.path(O,'analysis_data_private.csv'),check.names=FALSE,stringsAsFactors=FALSE)
set.seed(20261103); B<-2000L
clinical<-c('age','male','smoking','hypertension','diabetes','coronary_disease','ldl_c')
numeric_us<-c('cca_psv','ica_psv','thickness','length','gsm','jba')
cat_us<-c('stenosis','echo','homogeneity','surface','ulcer','calcification')
for(c in c(cat_us,'stenosis3','stenosis4'))d[[c]]<-factor(d[[c]])
full<-c(clinical,'stenosis3','antiplatelet','lipid_lowering')
Xus<-model.matrix(reformulate(c(numeric_us,cat_us)),d)[,-1,drop=FALSE]
Xcl<-as.matrix(d[,clinical]);Xextra<-model.matrix(~stenosis3+antiplatelet+lipid_lowering,d)[,-1,drop=FALSE]
# All indistinguishable baseline profiles share folds, preventing duplicate-profile leakage.
profile_cols<-c(clinical,'ai_probability',numeric_us,cat_us,'antiplatelet','lipid_lowering')
profile<-apply(d[,profile_cols],1,paste,collapse='|');grp<-match(profile,unique(profile))
folds<-function(ids,k,seed){
 set.seed(seed);gs<-unique(grp[ids]);f<-integer(length(ids))
 strata<-sapply(gs,function(g)any(d$event[ids[grp[ids]==g]]==1))
 for(s in c(FALSE,TRUE)){gg<-sample(gs[strata==s]);zz<-rep(seq_len(k),length.out=length(gg));f[match(grp[ids],gg,nomatch=0)>0]<-zz[match(grp[ids][match(grp[ids],gg,nomatch=0)>0],gg)]}
 stopifnot(all(f>0));f
}
outer<-folds(seq_len(nrow(d)),5,20261103)
# G(t-) for cases and G(h) for controls. Administrative censoring at 36 is excluded.
weights<-function(train,test,h){
 z<-d[train,];s<-survfit(Surv(time_months,as.integer(event==0 & time_months<36))~1,z)
 G<-function(t,left=FALSE){vapply(t,function(v){ix<-which(if(left)s$time<v else s$time<=v);if(!length(ix))1 else s$surv[max(ix)]},0.)}
 zz<-d[test,];y<-as.integer(zz$event==1 & zz$time_months<=h);w<-rep(0,length(test))
 w[y==1]<-1/pmax(G(zz$time_months[y==1],TRUE),1e-6)
 control<-y==0 & zz$time_months>=h;w[control]<-1/pmax(G(rep(h,sum(control))),1e-6)
 list(y=y,w=w)
}
loss<-function(y,p,w)sum(w*(-y*log(pmax(p,1e-8))-(1-y)*log(pmax(1-p,1e-8))))/sum(w)
lams<-10^seq(1,-4,length.out=6)
fitpath<-function(X,ids,h){q<-weights(ids,ids,h);ok<-q$w>0;glmnet(X[ids[ok],,drop=FALSE],q$y[ok],family='binomial',alpha=0,lambda=lams,weights=q$w[ok],standardize=TRUE)}
ridge<-function(X,tr,te,h,seed){
 ff<-folds(tr,3,seed);ll<-matrix(0,3,length(lams))
 for(j in 1:3){a<-tr[ff!=j];b<-tr[ff==j];f<-fitpath(X,a,h);p<-predict(f,X[b,,drop=FALSE],s=lams,type='response');q<-weights(a,b,h);ll[j,]<-apply(p,2,function(v)loss(q$y,v,q$w))}
 lam<-lams[which.min(colMeans(ll))];f<-fitpath(X,tr,h)
 list(p=as.numeric(predict(f,X[te,,drop=FALSE],s=lam,type='response')),lambda=lam)
}
logit<-function(p)qlogis(pmin(pmax(p,1e-6),1-1e-6))
us_stage<-function(tr,te,h,seed){
 f<-folds(tr,3,seed);u<-rep(NA_real_,length(tr))
 for(j in 1:3)u[f==j]<-ridge(Xus,tr[f!=j],tr[f==j],h,seed+j)$p
 v<-ridge(Xus,tr,te,h,seed+100)$p
 list(train=logit(u),test=logit(v))
}
stack_matrices<-function(u,tr,te,extra,ct){
 a<-cbind(Xcl[tr,,drop=FALSE],us=u$train);b<-cbind(Xcl[te,,drop=FALSE],us=u$test)
 if(extra){a<-cbind(a,Xextra[tr,,drop=FALSE]);b<-cbind(b,Xextra[te,,drop=FALSE])}
 if(ct){a<-cbind(a,ct=d$ai_probability[tr]);b<-cbind(b,ct=d$ai_probability[te])}
 list(a=a,b=b)
}
stack_ridge<-function(tr,te,h,seed){
 ff<-folds(tr,3,seed);us<-us_stage(tr,te,h,seed+50);cache<-list()
 for(j in 1:3){a<-tr[ff!=j];b<-tr[ff==j];cache[[j]]<-list(a=a,b=b,u=us_stage(a,b,h,seed+j*200))}
 res<-list()
 for(extra in c(FALSE,TRUE))for(ct in c(FALSE,TRUE)){
  ll<-matrix(0,3,length(lams))
  for(j in 1:3){z<-cache[[j]];xx<-stack_matrices(z$u,z$a,z$b,extra,ct);q<-weights(z$a,z$a,h);ok<-q$w>0
   f<-glmnet(xx$a[ok,,drop=FALSE],q$y[ok],weights=q$w[ok],family='binomial',alpha=0,lambda=lams,standardize=TRUE)
   pp<-predict(f,xx$b,s=lams,type='response');qt<-weights(z$a,z$b,h);ll[j,]<-apply(pp,2,function(p)loss(qt$y,p,qt$w))}
  lam<-lams[which.min(colMeans(ll))];xx<-stack_matrices(us,tr,te,extra,ct);q<-weights(tr,tr,h);ok<-q$w>0
  f<-glmnet(xx$a[ok,,drop=FALSE],q$y[ok],weights=q$w[ok],family='binomial',alpha=0,lambda=lams,standardize=TRUE)
  key<-paste0('Clinical_US',if(extra)'_stenosis_treatment' else '',if(ct)'_CT' else '')
  res[[key]]<-as.numeric(predict(f,xx$b,s=lam,type='response'))
 }
 res
}
P<-list();tuning<-list()
for(h in c(12,24,36)){
 cat('HORIZON',h,'\n');flush.console()
 pp<-data.frame(AI_probability=d$ai_probability,AI_classification=d$ai_binary,US_continuous=NA_real_,Clinical=NA_real_,Clinical_AI_probability=NA_real_,Clinical_AI_classification=NA_real_,Clinical_US=NA_real_,Clinical_US_CT=NA_real_,Clinical_US_stenosis_treatment=NA_real_,Clinical_US_stenosis_treatment_CT=NA_real_,Clinical_stenosis_treatment=NA_real_,Clinical_stenosis_treatment_CT=NA_real_)
 for(k in 1:5){
  tr<-which(outer!=k);te<-which(outer==k);seed<-20261103+h+k
  for(key in c('US_continuous','Clinical','Clinical_AI_probability','Clinical_AI_classification','Clinical_stenosis_treatment','Clinical_stenosis_treatment_CT')){
   X<-switch(key,US_continuous=Xus,Clinical=Xcl,Clinical_AI_probability=cbind(Xcl,ct=d$ai_probability),Clinical_AI_classification=cbind(Xcl,ct=d$ai_binary),Clinical_stenosis_treatment=cbind(Xcl,Xextra),Clinical_stenosis_treatment_CT=cbind(Xcl,Xextra,ct=d$ai_probability))
   z<-ridge(X,tr,te,h,seed);pp[te,key]<-z$p;tuning[[length(tuning)+1]]<-data.frame(horizon=h,fold=k,model=key,lambda=z$lambda)
  }
  ss<-stack_ridge(tr,te,h,seed+500);for(key in names(ss))pp[te,key]<-ss[[key]]
  cat(' fold',k,'done\n');flush.console()
 }
 stopifnot(!anyNA(pp));P[[as.character(h)]]<-pp
 write.csv(cbind(d[,c('patient_id','source','event','time_months')],outer_fold=outer,pp),file.path(O,paste0('OOF_predictions_',h,'m_private.csv')),row.names=FALSE)
}
write.csv(do.call(rbind,tuning),file.path(O,'Ridge_tuning.csv'),row.names=FALSE)
saveRDS(list(P=P,outer=outer,d=d),file.path(O,'analysis_predictions.rds'))
# Weighted empirical cumulative/dynamic AUC, ties receive half credit.
auc<-function(y,p,w){ord<-order(p);a<-w[ord]*y[ord];b<-w[ord]*(1-y[ord]);g<-match(p[ord],unique(p[ord]));aa<-as.numeric(rowsum(a,g));bb<-as.numeric(rowsum(b,g));sum(aa*(cumsum(bb)-bb+bb/2))/(sum(aa)*sum(bb))}
brier<-function(y,p,w)mean(w*(y-p)^2)
metrics<-list();diffs<-list();cal<-list();roc<-list();decision<-list()
families<-list(CT_vs_US=c('US_continuous','AI_probability'),Clinical_CT_vs_US=c('Clinical_US','Clinical_AI_probability'),Increment_clinical_US=c('Clinical_US','Clinical_US_CT'),Increment_US_stenosis_treatment=c('Clinical_US_stenosis_treatment','Clinical_US_stenosis_treatment_CT'),Increment_clinical_stenosis_treatment=c('Clinical_stenosis_treatment','Clinical_stenosis_treatment_CT'))
for(h in c(12,24,36)){
 pp<-P[[as.character(h)]];q<-weights(seq_len(nrow(d)),seq_len(nrow(d)),h);av<-sapply(pp,function(p)auc(q$y,p,q$w));bv<-sapply(pp,function(p)brier(q$y,p,q$w));ab<-bb<-matrix(NA_real_,B,ncol(pp));colnames(ab)<-colnames(bb)<-names(pp)
 set.seed(3011+h)
 for(i in seq_len(B)){
  ix<-sample.int(nrow(d),replace=TRUE);qt<-weights(ix,ix,h)
  ab[i,]<-sapply(pp,function(p)auc(qt$y,p[ix],qt$w));bb[i,]<-sapply(pp,function(p)brier(qt$y,p[ix],qt$w))
 }
 for(key in names(pp)){
  p<-pp[[key]];ci<-quantile(ab[,key],c(.025,.975));bc<-quantile(bb[,key],c(.025,.975));ok<-q$w>0;lg<-logit(p)
  f<-suppressWarnings(glm(q$y[ok]~lg[ok],weights=q$w[ok],family=quasibinomial()));f0<-suppressWarnings(glm(q$y[ok]~1+offset(lg[ok]),weights=q$w[ok],family=quasibinomial()))
  cs<-coef(summary(f));c0<-coef(summary(f0));
  metrics[[length(metrics)+1]]<-data.frame(horizon=h,model=key,n=nrow(d),events=sum(q$y),AUC=av[key],AUC_low=ci[1],AUC_high=ci[2],Brier=bv[key],Brier_low=bc[1],Brier_high=bc[2],CIL=coef(f0)[1],CIL_low=coef(f0)[1]-1.96*c0[1,2],CIL_high=coef(f0)[1]+1.96*c0[1,2],slope=coef(f)[2],slope_low=coef(f)[2]-1.96*cs[2,2],slope_high=coef(f)[2]+1.96*cs[2,2])
  ord<-order(p,decreasing=TRUE);ww<-q$w[ord];yy<-q$y[ord];roc[[length(roc)+1]]<-data.frame(horizon=h,model=key,FPR=c(0,cumsum(ww*(1-yy))/sum(ww*(1-yy))),TPR=c(0,cumsum(ww*yy)/sum(ww*yy)))
  bins<-pmin(5,ceiling(rank(p,ties.method='first')/length(p)*5));for(g in 1:5){ix<-which(bins==g);sf<-survfit(Surv(time_months,event)~1,d[ix,]);v<-summary(sf,times=h,extend=TRUE);cal[[length(cal)+1]]<-data.frame(horizon=h,model=key,bin=g,predicted=mean(p[ix]),observed=1-v$surv,low=1-v$upper,high=1-v$lower)}
  for(t in seq(.01,.15,.005)){hi<-p>=t;decision[[length(decision)+1]]<-data.frame(horizon=h,model=key,threshold=t,NB=mean(q$w*(q$y*hi-(1-q$y)*hi*t/(1-t))),Treat_all=mean(q$w*(q$y-(1-q$y)*t/(1-t))))}
 }
 for(fam in names(families)){keys<-families[[fam]];delta<-av[keys[2]]-av[keys[1]];dr<-ab[,keys[2]]-ab[,keys[1]];ci<-quantile(dr,c(.025,.975));db<-bb[,keys[2]]-bb[,keys[1]];bc<-quantile(db,c(.025,.975));pv<-(1+sum(abs(dr-delta)>=abs(delta)))/(B+1)
  diffs[[length(diffs)+1]]<-data.frame(horizon=h,family=fam,base=keys[1],expanded=keys[2],delta_AUC=delta,low=ci[1],high=ci[2],centered_bootstrap_P=pv,delta_Brier=bv[keys[2]]-bv[keys[1]],Brier_low=bc[1],Brier_high=bc[2])}
 cat('bootstrap',h,'done\n');flush.console()
}
m<-do.call(rbind,metrics);di<-do.call(rbind,diffs);di$Holm_P<-ave(di$centered_bootstrap_P,di$family,FUN=function(x)p.adjust(x,'holm'))
for(obj in c('m','di','cal','roc','decision')){z<-get(obj);if(is.list(z)&&!is.data.frame(z))z<-do.call(rbind,z);write.csv(z,file.path(O,paste0(switch(obj,m='Metrics',di='Paired_differences',cal='Calibration',roc='ROC',decision='DCA'),'.csv')),row.names=FALSE)}
# KM, risks and censor marks use survival methods, not crude percentages.
d$us_probability<-P[['36']]$US_continuous;cut<-median(d$us_probability);d$us_binary<-as.integer(d$us_probability>=cut)
write.csv(data.frame(US_OOF_median=cut),file.path(O,'US_cutpoint.csv'),row.names=FALSE)
kmrows<-list();riskrows<-list()
for(ex in c('ai_binary','us_binary')){
 sf<-survfit(Surv(time_months,event)~d[[ex]],d);lr<-survdiff(Surv(time_months,event)~d[[ex]],d);pv<-pchisq(lr$chisq,1,lower.tail=FALSE)
 s<-summary(sf,censored=TRUE);kmrows[[ex]]<-data.frame(assessment=ex,time=s$time,survival=s$surv,low=s$lower,high=s$upper,n_censor=s$n.censor,group=rep(c('Lower risk','Higher risk'),sapply(sf$strata,identity)),logrank_P=pv)
 # summary() may omit zero-time records; use its strata labels directly.
 kmrows[[ex]]$group<-ifelse(grepl('=0$',as.character(s$strata)),'Lower risk','Higher risk')
 for(g in 0:1)for(h in c(0,6,12,18,24,30,36)){
  sub<-d[d[[ex]]==g,];v<-summary(survfit(Surv(time_months,event)~1,sub),times=h,extend=TRUE)
  riskrows[[length(riskrows)+1]]<-data.frame(assessment=ex,group=if(g==0)'Lower risk' else 'Higher risk',horizon=h,n=nrow(sub),events=sum(sub$event==1&sub$time_months<=h),at_risk=sum(sub$time_months>=h),risk=1-v$surv,low=1-v$upper,high=1-v$lower,logrank_P=pv)
 }
}
km<-do.call(rbind,kmrows);rr<-do.call(rbind,riskrows);write.csv(km,file.path(O,'KM_coordinates.csv'),row.names=FALSE);write.csv(rr,file.path(O,'KM_horizon_risks.csv'),row.names=FALSE)
cols<-c('Lower risk'='#3480b8','Higher risk'='#c82423')
pngplot<-function(file,w,h,expr){agg_png(file.path(O,file),width=w,height=h,units='in',res=240);eval.parent(substitute(expr));dev.off()}
pngplot('Figure_6_updated.png',12,7,{
 par(mfrow=c(1,2),mar=c(6,4,3,1));for(ex in c('ai_binary','us_binary')){sf<-survfit(Surv(time_months,event)~d[[ex]],d);plot(sf,col=cols,lwd=2,mark.time=TRUE,conf.int=FALSE,xlim=c(0,36),ylim=c(.85,1),xlab='Months since baseline imaging',ylab='Event-free survival',main=if(ex=='ai_binary')'A  Frozen CT classification' else 'B  Continuous ultrasound risk groups');legend('bottomleft',names(cols),col=cols,lty=1,lwd=2,bty='n');mtext(paste('Log-rank P',format.pval(rr$logrank_P[rr$assessment==ex][1],digits=3,eps=.001)),side=3,line=.2,cex=.8);for(g in 0:1)mtext(paste(if(g==0)'Lower:' else 'Higher:',paste(rr$at_risk[rr$assessment==ex&rr$group==if(g==0)'Lower risk' else 'Higher risk'],collapse='  ')),side=1,line=3+g,cex=.65)}
})
# Robust full-period Cox summaries and period-specific analyses.
coxrows<-list();ph<-list();pw<-list();coefs<-list()
form<-function(v,response='Surv(time_months,event)')as.formula(paste(response,'~',paste(v,collapse='+')))
extract<-function(f,key,ex){s<-summary(f);j<-match(ex,rownames(s$coefficients));data.frame(model=key,exposure=ex,n=f$n,events=f$nevent,HR=s$conf.int[j,1],low=s$conf.int[j,3],high=s$conf.int[j,4],p=s$coefficients[j,'Pr(>|z|)'])}
specs<-list(M1=character(),M2=clinical,M3=c(clinical,'stenosis3'),M4=full)
for(ex in c('ai10','ai_binary'))for(key in names(specs)){
 f<-coxph(form(c(ex,specs[[key]])),d,ties='efron',x=TRUE,robust=TRUE);coxrows[[paste(ex,key)]]<-extract(f,key,ex);z<-cox.zph(f);ph[[paste(ex,key)]]<-data.frame(model=key,exposure=ex,term=rownames(z$table),z$table,row.names=NULL)
 if(key=='M4')coefs[[ex]]<-data.frame(exposure=ex,term=names(coef(f)),summary(f)$coefficients,row.names=NULL)
}
sp<-survSplit(Surv(time_months,event)~.,d,cut=c(12,24),episode='period',start='start')
for(ex in c('ai10','ai_binary'))for(key in c('Unadjusted','M4')){
 intervals<-1:3;if(ex=='ai_binary' && length(unique(d$ai_binary[d$event==1&d$time_months<=12]))==1)intervals<-2:3
 z<-sp[sp$period%in%intervals,];terms<-paste0('AI_period',intervals);for(i in intervals)z[[paste0('AI_period',i)]]<-z[[ex]]*(z$period==i)
 f<-coxph(form(c(terms,if(key=='M4')full,'strata(period)'),response='Surv(start,time_months,event)'),z,ties='efron',robust=TRUE,cluster=patient_id,x=TRUE)
 for(i in intervals){r<-extract(f,key,paste0('AI_period',i));r$exposure<-ex;r$period<-c('0 to 12','12 to 24','24 to 36')[i];pw[[length(pw)+1]]<-r}
}
write.csv(do.call(rbind,coxrows),file.path(O,'Adjusted_Cox.csv'),row.names=FALSE);write.csv(do.call(rbind,ph),file.path(O,'PH_tests.csv'),row.names=FALSE);write.csv(do.call(rbind,pw),file.path(O,'Piecewise_Cox.csv'),row.names=FALSE);write.csv(do.call(rbind,coefs),file.path(O,'Cox_covariates.csv'),row.names=FALSE)
subrows<-list();for(ex in c('ai10','ai_binary'))for(g in levels(d$stenosis3)){z<-d[d$stenosis3==g,];f<-coxph(form(c(ex,clinical,'antiplatelet','lipid_lowering')),z,ties='efron',robust=TRUE);r<-extract(f,'M4_stenosis_subgroup',ex);r$stenosis<-g;subrows[[length(subrows)+1]]<-r};write.csv(do.call(rbind,subrows),file.path(O,'Stenosis_subgroups.csv'),row.names=FALSE)
# Conditional IPCW sensitivity for CT discrimination, adjusting the censoring model.
cf<-coxph(form(c('ai_probability',full),response='Surv(time_months,as.integer(event==0 & time_months<36))'),d,ties='efron');bh<-basehaz(cf,centered=FALSE);lp<-predict(cf,type='lp',reference='zero');sen<-list()
for(h in c(12,24,36)){y<-as.integer(d$event==1&d$time_months<=h);ht<-ifelse(y==1,d$time_months,h);hz<-vapply(seq_len(nrow(d)),function(i){ix<-which(if(y[i]==1)bh$time<ht[i] else bh$time<=ht[i]);if(length(ix))bh$hazard[max(ix)] else 0},0.);gg<-exp(-hz*exp(lp));w<-ifelse(y==1|d$time_months>=h,1/pmax(gg,.01),0);sen[[length(sen)+1]]<-data.frame(horizon=h,AUC_conditional_IPCW=auc(y,d$ai_probability,w),max_weight=max(w),G_min=min(gg))};write.csv(do.call(rbind,sen),file.path(O,'Conditional_IPCW_sensitivity.csv'),row.names=FALSE)
# Loss-to-follow-up selection is described separately from the event outcome.
d$early_loss<-as.integer(d$event==0&d$time_months<36);bl<-list();for(v in c(clinical,'ai_probability','ai_binary','antiplatelet','lipid_lowering')){a<-d[d$early_loss==0,v];b<-d[d$early_loss==1,v];bl[[v]]<-data.frame(variable=v,not_early_loss_mean=mean(a),early_loss_mean=mean(b),SMD=(mean(b)-mean(a))/sqrt((var(a)+var(b))/2))};write.csv(do.call(rbind,bl),file.path(O,'Loss_baseline_comparison.csv'),row.names=FALSE)
saveRDS(list(d=d,P=P,metrics=m,differences=di),file.path(O,'final_analysis.rds'));capture.output(sessionInfo(),file=file.path(O,'R_session.txt'))
cat('COMPLETE\n');print(m[m$model=='AI_probability',]);print(di)
