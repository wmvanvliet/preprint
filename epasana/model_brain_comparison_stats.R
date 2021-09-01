library('lme4')
library('lmerTest')

data = read.csv('/m/nbe/scratch/reading_models/epasana/brain_model_comparison.csv')
model11 = lmer(LeftOcci1 ~ conv1_relu + (0 + conv1_relu | subject), data=data)
model12 = lmer(LeftOcci1 ~ conv2_relu + (0 + conv2_relu | subject), data=data)
model13 = lmer(LeftOcci1 ~ conv3_relu + (0 + conv3_relu | subject), data=data)
model14 = lmer(LeftOcci1 ~ conv4_relu + (0 + conv4_relu | subject), data=data)
model15 = lmer(LeftOcci1 ~ conv5_relu + (0 + conv5_relu | subject), data=data)
model16 = lmer(LeftOcci1 ~ fc1_relu + (0 + fc1_relu | subject), data=data)
model17 = lmer(LeftOcci1 ~ fc2_relu + (0 + fc2_relu | subject), data=data)
model18 = lmer(LeftOcci1 ~ word_relu + (0 + word_relu | subject), data=data)

model21 = lmer(LeftOcciTemp2 ~ conv1_relu + (0 + conv1_relu | subject), data=data)
model22 = lmer(LeftOcciTemp2 ~ conv2_relu + (0 + conv2_relu | subject), data=data)
model23 = lmer(LeftOcciTemp2 ~ conv3_relu + (0 + conv3_relu | subject), data=data)
model24 = lmer(LeftOcciTemp2 ~ conv4_relu + (0 + conv4_relu | subject), data=data)
model25 = lmer(LeftOcciTemp2 ~ conv5_relu + (0 + conv5_relu | subject), data=data)
model26 = lmer(LeftOcciTemp2 ~ fc1_relu + (0 + fc1_relu | subject), data=data)
model27 = lmer(LeftOcciTemp2 ~ fc2_relu + (0 + fc2_relu | subject), data=data)
model28 = lmer(LeftOcciTemp2 ~ word_relu + (0 + word_relu | subject), data=data)

model31 = lmer(LeftTemp3 ~ conv1_relu + (0 + conv1_relu | subject), data=data)
model32 = lmer(LeftTemp3 ~ conv2_relu + (0 + conv2_relu | subject), data=data)
model33 = lmer(LeftTemp3 ~ conv3_relu + (0 + conv3_relu | subject), data=data)
model34 = lmer(LeftTemp3 ~ conv4_relu + (0 + conv4_relu | subject), data=data)
model35 = lmer(LeftTemp3 ~ conv5_relu + (0 + conv5_relu | subject), data=data)
model36 = lmer(LeftTemp3 ~ fc1_relu + (0 + fc1_relu | subject), data=data)
model37 = lmer(LeftTemp3 ~ fc2_relu + (0 + fc2_relu | subject), data=data)
model38 = lmer(LeftTemp3 ~ word_relu + (0 + word_relu | subject), data=data)

results = 
	rbind(summary(model11)$coef[c('conv1_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model12)$coef[c('conv2_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model13)$coef[c('conv3_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model14)$coef[c('conv4_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model15)$coef[c('conv5_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model16)$coef[c('fc1_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model17)$coef[c('fc2_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model18)$coef[c('word_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model21)$coef[c('conv1_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model22)$coef[c('conv2_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model23)$coef[c('conv3_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model24)$coef[c('conv4_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model25)$coef[c('conv5_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model26)$coef[c('fc1_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model27)$coef[c('fc2_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model28)$coef[c('word_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model31)$coef[c('conv1_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model32)$coef[c('conv2_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model33)$coef[c('conv3_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model34)$coef[c('conv4_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model35)$coef[c('conv5_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model36)$coef[c('fc1_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model37)$coef[c('fc2_relu'), c('Estimate', 'Pr(>|t|)')],
          summary(model38)$coef[c('word_relu'), c('Estimate', 'Pr(>|t|)')])

results = cbind(c(rep('LeftOcci1', 8), rep('LeftOcciTemp2', 8), rep('LeftTemp3', 8)), results)
results = cbind(rep(c('conv1_relu', 'conv2_relu', 'conv3_relu', 'conv4_relu', 'conv5_relu', 'fc1_relu', 'fc2_relu', 'word_relu'), 3), results)
colnames(results) <- c('Model layer', 'Dipole group', 'Correlation', 'P value')

write.csv(results, 'results.csv')
