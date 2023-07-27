# FOSMix
Frequency-based Optimal Style Mix

## Step 0
Install packages
```
pip install -r requirements.txt
```

## Step 1
Modify the installed packages

```bash
./modify_package_contents.sh
```
## TODO
- 不要なコードの削除
- printからloggerに変更
 - modify_package._contents.shの動作確認
 - sourceディレクトリのファイル置き場のクリーン化
 - colabで動作確認


# コードの説明

## 引数の説明

*は必須項目

1. --dataset(str*) : データセットの選択   

        OEM (OpenEarthMap)
        FLAIR

2. --n_epochs(int*) : 学習エポック数

        150 (OEMデータセット)
        50 (FLAIRデータセット)

3. --ver(int*) : バージョン

4. --final(bool) : テストする際のモデルパラメータを最終的なパラメータにする

        0 (validationデータに対する結果が最も良いパラメータをテスト時に使用)
        1 (最後のパラメータを使用)

5. --randomize(bool*) : 画像をランダム化するかしないか、つまりbaselineを使用するか提案を使用するか

        0 (baseline)
        1 (何かしらの提案のパートを使用する)

6. --optimize(bool*) : マスクを最適化するか否か、`optimize`が1の時は必ず`randomize`が1となる

        0 (baseline or FULL MIX)
        1 (OPTIMAL MIX)

7. --aug_color(float) : Augmentationでカラー変化を行う確率

        from 0 to 1

8. --MFI (int*) : Mask from Image, OPTIMAL MASKを画像から生成するか否か

        0 (1つのマスクを全ての画像に対して学習し使用)
        1 (画像から生成)

9. --fullmask (int*) : FULL MIXを使用するか否か

        0 (FULL MIXを使用しない)
        1 (FULL MIXを使用)
