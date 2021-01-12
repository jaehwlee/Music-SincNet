Music Auto-tagging with Learning Filter Banks
===
* [KSC 2020 Paper](https://github.com/jaehwlee/music-sincnet/files/5760095/2020KSC_.pdf)

* [Video Instruction](https://www.youtube.com/watch?v=9oyLRW1R6kY&t=3s)

Summary
--
* SincNet을 활용하여 주어진 데이터에 적합한 필터 뱅크를 학습할 수 있는 End-to-End 모델
* MTAT 데이터 셋에서 AUC 0.9115로 원시 음원 파형을 입력으로 한 모델 중 가장 높은 성능을 보임

Model Architecture
--
![image](https://user-images.githubusercontent.com/33409264/103455059-ee7cfb00-4d2c-11eb-954b-25465d6b3a3d.png)

* 대역 필터 역할을 수행하기 위해 [SincNet](https://arxiv.org/abs/1808.00158)의 필터를 활용
* 필터 뱅크의 초기 주파수 최대값을 약 4kHz, 학습을 통해 최대 약 11kHz까지 늘어나도록 설정

Results
--
![image](https://user-images.githubusercontent.com/33409264/103455148-bf1abe00-4d2d-11eb-9f05-c4a19458e579.png)

* 원시 음원 파형을 입력으로 한 모델 중 가장 높은 성능을 보임

Analysis
--
![image](https://user-images.githubusercontent.com/33409264/103455177-0c972b00-4d2e-11eb-96d9-d721468662e9.png)

* Mel-scale 필터 뱅크는 주파수 대역이 높아짐에 따라 필터의 대역폭이 넓어지고, 로그 함수적으로 진폭 감소
* Musinc-Sinc 필터 뱅크는 2.5kHz, 3.8kHz 대역의 진폭이 타 대역보다 큼
* 2.5kHz의 바이올린, 일렉 기타와 3.8kHz의 킥 드럼 등이 태깅에 영향을 미친 주요 특징 중에 하나
