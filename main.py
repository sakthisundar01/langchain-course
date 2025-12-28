from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    Manchester United Football Club, commonly referred to as Man United (often stylised as Man Utd) or simply United, is a professional football club based in Old Trafford, Greater Manchester, England. They compete in the Premier League, the top tier of English football. Nicknamed the Red Devils, they were founded as Newton Heath LYR Football Club in 1878, but changed their name to Manchester United in 1902. After a spell playing in Clayton, Manchester, the club moved to their current stadium, Old Trafford, in 1910.

Domestically, Manchester United have won a joint-record twenty top-flight league titles, thirteen FA Cups, six League Cups and a record twenty-one FA Community Shields. Additionally, in international football, they have won the European Cup/UEFA Champions League three times, and the UEFA Europa League, the UEFA Cup Winners' Cup, the UEFA Super Cup, the Intercontinental Cup and the FIFA Club World Cup once each.[7][8] Appointed as manager in 1945, Matt Busby built a team with an average age of just 22 nicknamed the Busby Babes that won successive league titles in the 1950s and became the first English club to compete in the European Cup. Eight players were killed in the Munich air disaster, but Busby rebuilt the team around star players George Best, Denis Law and Bobby Charlton – known as the United Trinity. They won two more league titles before becoming the first English club to win the European Cup in 1968.

After Busby's retirement, Manchester United were unable to produce sustained success until the arrival of Alex Ferguson, who became the club's longest-serving and most successful manager, winning 38 trophies including 13 league titles, five FA Cups and two Champions League titles between 1986 and 2013.[9] In the 1998–99 season, under Ferguson, the club became the first in the history of English football to achieve the continental treble of the Premier League, FA Cup and UEFA Champions League.[10] In winning the UEFA Europa League under José Mourinho in 2016–17, they became one of five clubs to have won the original three main UEFA club competitions (the Champions League, Europa League and Cup Winners' Cup).
    """

    summary_template = """
    given in the information {information} about the story of Manchester United I want you to create:
    1. A Short Story
    2. Two Intersting plots about the story
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()