- 이 디렉토리에서 다음과 같이 위키피디아 덤프를 파싱하면 **text** 디렉토리가 생성됩니다.
- **text** 디렉토리에는 AA, AB, AC ... 등의 디렉토리가 있고 여기에 한국어 위키 텍스트가 나뉘어 담겨 있습니다.   
```
cd wikidump
wikiextractor kowiki-latest-pages-articles.xml.bz2 --no-templates
```
