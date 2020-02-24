library(tidyverse)
library(readxl)
library(factoextra)
library(tidytext)
library(wordcloud)
library(quanteda)
library(sentimentr)
library(topicmodels)
library(cluster)

library(streamgraph)
library(plotly)


##read data
raw_review<-readRDS('datasets/hw2.rds')
View(raw_review)
skimr::skim(raw_review) #no missing value; (18555,9)

##########EDA##########

#rating for each department
raw_review %>% group_by(department_name) %>% 
  ggplot(aes(x=department_name,y=rating,color=department_name)) +
  geom_boxplot(show.legend = F) +
  ggtitle('Rating for Departments')   ##department 6 shows the worst performance among all deps.
#rating for each division
raw_review %>% group_by(division_name) %>% 
  ggplot(aes(x=division_name,y=rating,color=division_name)) +
  geom_boxplot(show.legend = F) +
  ggtitle('Rating for Divisions')  ## 3 divisions are same
#rating for each class
raw_review %>% group_by(class_name) %>% 
  ggplot(aes(x=class_name,y=rating,color=class_name)) +
  geom_boxplot(show.legend = F) +
  ggtitle('Rating for Classes')  ## class 18 and 20 obviously underperform compared with others 
#how many products under each department
uni_pdt_dep<-raw_review %>% group_by(department_name) %>% 
  summarise(uni_pdt=length(unique(product_id)))

#how many products under each division
uni_pdt_div<-raw_review %>% group_by(division_name) %>% 
  summarise(uni_pdt=length(unique(product_id)))

#how many products under each class
uni_pdt_class<-raw_review %>% group_by(class_name) %>% 
  summarise(uni_pdt=length(unique(product_id)))  #18 and 20  

#########sentiment analysis##########

#unnest text
review<-raw_review %>% select(c(crmid,review_text))

review_tidy<-review %>% unnest_tokens(token,review_text,token='words',strip_punct=T,strip_num=T)
head(review_tidy,20)
#remove stop words
sw=get_stopwords()
review_tidy_nosw<-review_tidy %>% anti_join(sw,by=c('token'='word')) 
View(review_tidy_nosw)


############
#get sentiment for non-stopwords: sentiment r (sentence base)
text<-review$review_text %>% get_sentences() %>% sentiment()
head(text,20)

senti_afi_2<-text %>% group_by(element_id) %>% summarise(polarity=mean(sentiment))
nrow(senti_afi_2)==nrow(raw_review)

#what is it overall
skimr::skim(senti_afi_2)
#join onto the original data
review_senti2 = raw_review %>% mutate(polarity=senti_afi_2$polarity)
View(review_senti2)
#visualization for interesting patterns
length(unique(raw_review$product_id))   #1083/18555
unique(raw_review$division_name)  #1 2 3
unique(raw_review$department_name)  #1-6


  #distribution of the sentiment overall
ggplot(review_senti2, aes(x=polarity)) + geom_histogram(fill='black') +ggtitle('Distribution Plot of Polarity Scores')
quantile(review_senti2$polarity,probs = c(0.25,0.5,0.75))
postive_senti<-nrow(filter(review_senti2,polarity>0))/nrow(review_senti2)  # 0.8880086
  #sentiment vs age groups

          #ggplot(review_senti2,aes(x=age,y=polarity)) + geom_smooth()
review_senti2 %>% group_by(age) %>% summarise(mean_polarity=mean(polarity)) %>% 
  ggplot(aes(x=age,y=mean_polarity)) + geom_smooth() +
  ggtitle('Sentiment Analysis for Ages')


  #sentiment vs devision
ggplot(review_senti2,aes(x=division_name,y=polarity)) +
  geom_boxplot(aes(color=division_name),show.legend = F) +
  ggtitle('Sentiment Analysis for Divisions')

  #sentiment vs department
ggplot(review_senti2,aes(x=department_name,y=polarity)) +
  geom_boxplot(aes(color=department_name),show.legend = F)+
  ggtitle('Sentiment Analysis for Departments')

  #sentiment vs class
ggplot(review_senti2,aes(x=class_name,y=polarity)) +
  geom_boxplot(aes(color=class_name),show.legend = F)+
  ggtitle('Sentiment Analysis for Classes')


#########topic modelling##########
review_cps<-corpus(review$review_text)
summary(review_cps)
docvars(review_cps,'crmid')<-review$crmid
review_dfm<-dfm(review_cps,
                remove=get_stopwords()$word,
                remove_punct=T,
                remove_numbers=T,
                remove_symbols=T,
                remove_twitter=T,
                remove_url=T) %>% 
  dfm_trim(min_termfreq = 10,
           termfreq_type = 'count',
           max_docfreq = 0.7,
           docfreq_type = 'prop')
review_m<-convert(review_dfm,'topicmodels')


review_lda<-LDA(review_m,k=6,control = list(seed=820))
summary(review_lda)
#view beta
review_beta = tidy(review_lda, matrix="beta")
head(review_beta,30)

review_top20 = review_beta %>% 
  group_by(topic) %>% 
  top_n(20, beta) %>% 
  ungroup() %>% 
  arrange(topic,desc(beta)) 
review_top20

#view tokens example for each topic
terms(review_lda,20)

review_top20 %>% mutate(term=reorder(term,beta)) %>% 
  ggplot(aes(x=term,y=beta,fill=factor(topic))) +
  geom_col(show.legend = F)+
  facet_wrap(~topic,scales='free') +
  coord_flip()


#view gamma
review_gamma = tidy(review_lda, matrix="gamma")
review_gamma %>% 
  arrange(-gamma) %>% 
  print(n=30)


skimr::skim(fgamma)

ggplot(review_gamma,aes(x=gamma)) +
  geom_histogram() +
  facet_wrap(~topic)

   #review_gamma %>% group_by(document) %>% summarise(sum=sum(gamma))
##here is the loop to choose the optimal k#########
n_topic<-c()
n_doc<-c()
opt_k<-data.frame(n_topic=n_topic,n_doc=n_doc)
for (i in 2:10) {
  review_lda=LDA(review_m,k=i,control = list(seed=820))
  review_gamma = tidy(review_lda, matrix="gamma")
  ndoc=review_gamma %>% 
    arrange(-gamma) %>% 
    filter(gamma>1/i) %>% nrow()/nrow(review_gamma)
  opt_k[i-1,1]=i
  opt_k[i-1,2]=ndoc
}

opt_k


#######################################draft ############################################
#extend stopwords 

sw<-c(get_stopwords()$word,'dress')
review_dfm<-dfm(review_cps,
                remove=sw,
                remove_punct=T,
                remove_numbers=T,
                remove_symbols=T,
                remove_twitter=T,
                remove_url=T) %>% 
  dfm_trim(min_termfreq = 10,
           termfreq_type = 'count',
           max_docfreq = 0.7,
           docfreq_type = 'prop')
review_m<-as.matrix(review_dfm)

##############
#get sentiment for non-stopwords: bing(only 'positive' and 'negative' for each token)
review_tidy_sen_bing<-review_tidy_nosw %>% inner_join(get_sentiments('bing'),by=c('token'='word'))
head(review_tidy_sen_bing,10)

senti_bing<-review_tidy_sen_bing %>% count(crmid,token,sentiment) %>% 
  pivot_wider(names_from =sentiment ,values_from = n,values_fill = list(n=0)) %>% 
  group_by(crmid) %>% summarise(pos=sum(positive),neg=sum(negative),polarity=pos-neg) 

#what is it overall
skimr::skim(senti_bing)
#join onto the original data
review_senti1 = inner_join(review,senti_bing,by = "crmid")
View(review_senti1)
#ggplot(review_senti1, aes(x=airlines, y=polarity)) + geom_boxplot() 



