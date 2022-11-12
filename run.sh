for lr in {1,0.1,0.01,0.001}
do
  for alpha in {0.0001,0.001,0.1,0}

    do
        python main.py --model BOW-count --lr ${lr} --alpha ${alpha} --keep_topN 10000
        python main.py --model BOW-tfidf --lr ${lr} --alpha ${alpha} --keep_topN 10000
        python main.py --model BOCN-count --lr ${lr} --alpha ${alpha} --keep_topN 10000
        python main.py --model BOCN-tfidf --lr ${lr} --alpha ${alpha} --keep_topN 10000
        python main.py --model BOWBOCN-count --lr ${lr} --alpha ${alpha} --keep_topN 10000
        python main.py --model BOWBOCN-tfidf --lr ${lr} --alpha ${alpha} --keep_topN 10000
    done

done