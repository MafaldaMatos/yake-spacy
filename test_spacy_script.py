from spacy_script import eraseOverlapIntervals, load_annotated_data
import re
import json
from pathlib import Path
from src.models.spacy import SpacyModel
from src.data.spacy import SpacyDataLoader
import spacy
from tqdm import tqdm


ROOT = Path(__file__).parent.parent
ANNOTATED_DATA_PATH = ROOT / "data" / "annotated"


def test_eraseOverlapIntervals():
    result = eraseOverlapIntervals([[1,2], [3,4], [1,3]])
    expected_result = [[1, 2], [3,4]]
    assert result == expected_result


def test_eraseOverlapIntervals_2():
    result = eraseOverlapIntervals([[1227,1249], [1243,1249]])
    expected_result = [[1227, 1249]]
    print(result == expected_result)
    assert result == expected_result


def test_eraseOverlapIntervals_3():
    result = eraseOverlapIntervals( [[2, 24], [28, 43], [54, 62], [72, 79], [84, 99], [104, 116], [126, 131], [132, 141], [146, 164], [182, 207], [210, 216], [325, 331], [412, 424], [527, 533], [551, 572], [573, 579], [856, 871], [878, 883], [884, 893], [1067, 1073], [1077, 1086], [1227, 1249], [1243, 1249], [1293, 1299], [1414, 1423], [1439, 1448], [1471, 1477], [1582, 1588]]
)
    expected_result = [[2, 24], [28, 43], [54, 62], [72, 79], [84, 99], [104, 116], [126, 131], [132, 141], [146, 164], [182, 207], [210, 216], [325, 331], [412, 424], [527, 533], [551, 572], [573, 579], [856, 871], [878, 883], [884, 893], [1067, 1073], [1077, 1086], [1227, 1249], [1293, 1299], [1414, 1423], [1439, 1448], [1471, 1477], [1582, 1588]]

    print(result == expected_result)
    assert result == expected_result


def test_eraseOverlapIntervals_empty():
    result = eraseOverlapIntervals([])
    expected_result = []
    assert result == expected_result


def test_eraseOverlapIntervals_4():
    result = eraseOverlapIntervals( [[63, 85], [90, 101], [118, 124], [278, 284], [296, 309], [320, 343], [393, 399], [400, 404], [458, 464], [573, 579], [724, 730], [808, 814], [842, 848], [896, 915], [911, 935], [955, 961], [980, 986], [1112, 1118], [1347, 1353], [1356, 1362], [1364, 1375], [1604, 1610], [1795, 1801], [2126, 2132], [2509, 2515], [2561, 2567], [2590, 2596], [2708, 2714], [3051, 3072]]
    )
    expected_result = [[63, 85], [90, 101], [118, 124], [278, 284], [296, 309], [320, 343], [393, 399], [400, 404], [458, 464], [573, 579], [724, 730], [808, 814], [842, 848], [896, 915], [955, 961], [980, 986], [1112, 1118], [1347, 1353], [1356, 1362], [1364, 1375], [1604, 1610], [1795, 1801], [2126, 2132], [2509, 2515], [2561, 2567], [2590, 2596], [2708, 2714], [3051, 3072]]
    assert result == expected_result


def test_eraseOverlapIntervals_5():
    result = eraseOverlapIntervals([[296, 317], [338, 360], [393, 417], [300, 317], [344, 360], [296, 309], [338, 351], [393, 404], [458, 469], [904, 915], [980, 991], [2509, 2520], [896, 915], [400, 417], [278, 291], [808, 821], [955, 968], [320, 343], [118, 124], [724, 730], [842, 848], [1112, 1118], [1347, 1353], [1604, 1610], [2126, 2132], [2561, 2567], [2590, 2596], [2708, 2714], [393, 399], [458, 464], [904, 910], [980, 986], [2509, 2515], [400, 404], [465, 469], [911, 915], [987, 991], [2516, 2520], [90, 101], [1364, 1375], [3061, 3072], [3051, 3072], [911, 935], [63, 85], [278, 284], [573, 579], [808, 814], [955, 961], [1356, 1362], [1795, 1801]]
)
    expected_result = [[63, 85], [90, 101], [118, 124], [278, 284], [296, 309], [320, 343], [344, 360], [393, 399], [400, 404], [458, 464], [465, 469], [573, 579], [724, 730], [808, 814], [842, 848], [896, 915], [911, 915], [955, 961], [980, 986], [987, 991], [1112, 1118], [1347, 1353], [1356, 1362], [1364, 1375], [1604, 1610], [1795, 1801], [2126, 2132], [2509, 2515], [2516, 2520], [2561, 2567], [2590, 2596], [2708, 2714], [3051, 3072]]
    assert result == expected_result


def test_annotation():
    result = ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'L-TIMEX', 'O', 'O', 'B-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', '-', '-', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'I-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-TIMEX', 'L-TIMEX', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    ind = result.index('-')
    print(ind)

    text = '''Durante uma cimeira em Bruxelas, os representantes da Rússia e da União Europeia decidiram reunir-se uma vez por mês para discutir a crise internacional, uma decisão qualificada por Putin como "um importante passo em frente", informa a AFP.Numa declaração conjunta publicada durante a cimeira, a União Europeia e a Rússia exprimiram ser seu objectivo "intensificar a cooperação com o objectivo de combater o terrorismo", o que passa, nomeadamente, pela discussão das formas de troca de informações sobre redes terroristas ou sobre pessoas a elas ligadas.Os novos parceiros pretendem igualmente aumentar a sua colaboração no que diz respeito à venda de armas e explosivos ou ainda sobre as transacções financeiras susceptíveis de servir "actividades terroristas".No âmbito desta anunciada cooperação, Moscovo e os Quinze prometem igualmente trocar informações sobre "as novas formas de actividade terrorista", como "as ameaças químicas, biológicas ou nucleares".Questionado pelos jornalistas sobre a situação na república independentista da Tchetchénia, Putin acusou os guerrilheiros tchetchenos de serem financiados e armados por "centros de terrorismo internacional" onde os seus comandos "são preparados para agir em território russo". O Presidente russo estabeleceu mesmo um paralelismo entre os ataques de 11 de Setembro contra Washington e Nova Iorque e os atentados terroristas que em 1999 atingiram a capital russa, adiantando que ambos "tiveram a mesma mão dos terroristas internacionais".Durante a cimeira, a UE e Moscovo chegaram a acordo quanto à necessidade do alargamento da União Europeia "não criar novas linhas de separação na Europa". Neste âmbito, as duas partes propuseram que até ao final do ano seja encontrada uma solução para o enclave russo de Kaliningrado, que ficará rodeado pelos países da União, assim que esteja completo o alargamento da UE.
Putin aproveitou a sua deslocação a Bruxelas para voltar a mostrar maior disponibilidade em negociar com a NATO. Um pouco antes do encontro com George Robertson, secretário-geral da NATO, Putin afirmou que poderá rever a sua posição sobre o alargamento da instituição caso a Aliança se transforme numa organização menos militarista.Putin sublinhou, por várias vezes, a hostilidade russa à perspectiva de uma nova vaga de alargamento da NATO, que poderá ocorrer por ocasião da Cimeira da Aliança Atlântica, que se irá realizar em Novembro do próximo ano em Praga. "No que diz respeito ao alargamento da NATO, poderemos vir a olhar esse problema de uma maneira totalmente diferente, se se concretizarem as ideias que discutimos com a Europa, o mesmo é dizer caso a NATO se transforme, caso a organização se torne mais política", afirmou.'''
    res = re.findall(r'\S+', text)
    print(res[ind:len(text)-1])

    ind2 = text.index('sobre "as')
    print(ind2)


def test_re():
    keyword = "gabinete"
    text = '''O responsável faz um balanço positivo de todo o trabalho que até agora foi realizado, garantindo que a abertura deste tipo de equipamentos é totalmente justificada. Se por um lado, os gabinetes médico-legais não resolvem todos os problemas, por outro, há questões que ficam solucionadas. "Não resolve todos os problemas, mas garante a dignidade das perícias e intimidade às pessoas", explica. As "condições de frio para os cadáveres" foi um dos passos dados. No entanto, a grande questão continua a ser a falta de médicos para este tipo de trabalho. Neste momento, só 11 por cento do quadro do Instituto Nacional de Medicina Legal é que se encontra preenchido, existindo pouco mais de 30 médicos para mais de 200 lugares existentes. Duarte Nuno Vieira sabe que é necessário "dinamizar a qualificação dos médicos para trabalhar com mais qualidade", mesmo assim, salienta, "temos de reconhecer o esforço dos médicos". "O país tem de estar grato por fazerem o que fazem dentro das limitações que têm", defende. De qualquer forma, um dos próximos objectivos é exactamente promover essa formação junto da classe médica. 
Inicialmente estavam no papel 30 gabinetes médico-legais, só que a entrada em funcionamento do novo Hospital da Covilhã e do curso de medicina na Universidade de Beira Interior, também na Covilhã, levou à criação de mais um equipamento, no sentido de dar apoio às duas entidades. "O nosso gabinete não estava inicialmente previsto, mas era fundamental", conta Carlos Abreu, responsável pelo gabinete médico-legal da Covilhã, a funcionar no novo estabelecimento hospitalar, mas completamente independente do próprio hospital. As razões da sua abertura prenderam-se ainda com a cobertura de uma zona geográfica considerável, a área de circunscrição judicial, e que compreende os concelhos da Covilhã, Fundão e Sabugal. E, por vezes, é preciso dar apoio a outras comarcas que estão ali ao pé, como é o caso da Guarda, quando o gabinete médico-legal mais próximo de toda a região está situado em Coimbra. Segundo Carlos Abreu, o gabinete da Covilhã "tem condições extraordinárias" e está a funcionar em pleno, desde 1 de Julho, com o "mais moderno que há" e o mais funcional, salientando que para isso contribuiu o "empenhamento" do Instituto Nacional de Medicina Legal. Santa Maria da Feira não pode dizer o mesmo, pois há mais de dois anos que espera pela abertura de um gabinete médico-legal, depois da abertura do Hospital S. Sebastião, em 1999. Segundo fonte judicial, há "uma situação de ruptura, em termos de perícia" e "algumas dificuldades não justificáveis", quando o único problema que parecia existir estaria na realização de pequenas obras a efectuar na estrutura hospitalar da Feira. O que é certo é que se verificou um aumento do serviço da própria comarca feirense, crescendo "exponencialmente, em termos de Ministério Público, devido à existência do hospital", constata a mesma fonte. No entanto, o assunto parece estar resolvido. Segundo Duarte Nuno Vieira, até ao final do ano Santa Maria da Feira terá um gabinete médico-legal. Algumas obras de adaptação já foram resolvidas e o "hospital tem sido particularmente receptivo". Entretanto, Chaves está prestes a inaugurar o seu equipamento. Seguir-se-á Viseu, no início do próximo ano.
'''
    matches = re.finditer(keyword, text)
    for match in matches:
        print(text[match.start()-20:match.end()+20])
        print(text[match.start():match.end()])
        print(match.span())


def table():
    textdata1, entitiesdata1, textdata2, entitiesdata2 = load_annotated_data()

    print("train")
    print(len(textdata1))
    count = 0
    for i in range(len(entitiesdata1)):
        count = count + len(entitiesdata1[i])
    print(count)

    print()
    print("validation")
    print(len(textdata2))
    count = 0
    for i in range(len(entitiesdata2)):
        count = count + len(entitiesdata2[i])
    print(count)


def check_model(model_path):
    model = SpacyModel("pt")
    model.load(ROOT / model_path)

    eval_data = SpacyDataLoader(
        ANNOTATED_DATA_PATH / r'validation1000.json')
    
    prediction = model.predict(eval_data.texts)
    count = 0
    count_size = 0
    count_samesize = 0
    for i in tqdm(range(len(prediction))):
        #print(prediction[i])
        print(i)
        print(eval_data.texts[i])
        if prediction[i] != eval_data.annotations[i]:
            print(f"{prediction[i]}")
            print("!=")
            print(f"{eval_data.annotations[i]}")
            print()
            for s,e in prediction[i]:
                print(eval_data.texts[i][s:e])
            print("!=")
            for s,e in eval_data.annotations[i]:
                print(eval_data.texts[i][s:e])
            count = count + 1
            if len(prediction[i]) == len(eval_data.annotations[i]):
                count_samesize = count_samesize + 1
            if len(prediction[i]) < len(eval_data.annotations[i]):
                count_size = count_size + 1
        print()
    print("predicted != annotated")
    print(f"wrongly predicted = {count}/{len(prediction)}")
    print()
    print(f"same size predictions = {count_samesize}/{len(prediction)}")
    print(f"more annotations than predictions = {count_size}/{len(prediction)}")
    print(f"less annotations than predictions = {len(prediction)-count_size}/{len(prediction)}")


def load_test_annotated_data():
    a1 = ANNOTATED_DATA_PATH / r'train1000_size.json'
    a2 = ANNOTATED_DATA_PATH / r'validation1000_size.json'

    f1 = open(a1)
    data1 = json.load(f1)

    textdata1 = []
    entitiesdata1 = []
    for datai in data1:
        textdata1.append(datai[0])
        entitiesdata1.append(datai[1]["entities"])

    f2 = open(a2)
    data2 = json.load(f2)

    textdata2 = []
    entitiesdata2 = []
    for datai in data2:
        textdata2.append(datai[0])
        entitiesdata2.append(datai[1]["entities"])

    #return textdata1[:700], entitiesdata1[:700], textdata2[:300], entitiesdata2[:300]
    return textdata1, entitiesdata1, textdata2, entitiesdata2


def check_annotation():
    textdata1, entitiesdata1, textdata2, entitiesdata2 = load_annotated_data()

    nlp = spacy.blank("pt")

    count = 1

    for i in range(len(textdata1)):
        offsets = spacy.training.offsets_to_biluo_tags(nlp.make_doc(textdata1[i]), entitiesdata1[i])

        if '-' in offsets:
            if count != 0:
                print(textdata1[i])
                print(entitiesdata1[i])
                print(offsets)

                for s, e, _ in entitiesdata1[i]:
                    print(textdata1[i][s:e])
                
                ind = offsets.index('-')
                print(ind)

                res = re.findall(r'\S+', textdata1[i])
                print(len(res))
                print(res[ind:len(textdata1[i])-1])

                #ind2 = textdata1[i].index()
                #print(ind2)
                break
        count = count+1
    print(count-1)
    print(len(textdata1))
