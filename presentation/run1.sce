write_codes = true;
active_buttons = 2;
begin;

picture {
  default_code = "fixation";
  bitmap {
		 filename = "fixation_cross.png";
  };
  x = 0; y = 0;
} fixation;

TEMPLATE "stimulus.tem" {
word file code;
"hjqc" "consonants_hjqc.png" 20;
"v^v^" "symbols_v^v^.png" 30;
"zjwhs" "consonants_zjwhs.png" 20;
"syaani" "word_syaani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "word_syaani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"siitos" "word_siitos.png" 10;
"^vod" "symbols_^vod.png" 30;
"svds" "symbols_svds.png" 30;
"ssods" "symbols_ssods.png" 30;
"uiva" "word_uiva.png" 10;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oinas" "word_oinas.png" 10;
"estyä" "word_estyä.png" 10;
"hormi" "word_hormi.png" 10;
"uivelo" "word_uivelo.png" 10;
"zdvv" "consonants_zdvv.png" 20;
"^svs^" "symbols_^svs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_^svs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dvsvos" "symbols_dvsvos.png" 30;
"zkwnr" "consonants_zkwnr.png" 20;
"ripsi" "word_ripsi.png" 10;
"hnxf" "consonants_hnxf.png" 20;
"cnwh" "consonants_cnwh.png" 20;
"lemu" "word_lemu.png" 10;
"v^s^vd" "symbols_v^s^vd.png" 30;
"syöpyä" "word_syöpyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_syöpyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pesula" "word_pesula.png" 10;
"juhta" "word_juhta.png" 10;
"nisä" "word_nisä.png" 10;
"voo^o" "symbols_voo^o.png" 30;
"osdod^" "symbols_osdod^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s _" "symbols_osdod^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"luusto" "word_luusto.png" 10;
"eliö" "word_eliö.png" 10;
"sysi" "word_sysi.png" 10;
"^ooso" "symbols_^ooso.png" 30;
"rapsi" "word_rapsi.png" 10;
"tuohus" "word_tuohus.png" 10;
"rmmrh" "consonants_rmmrh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_rmmrh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"piip" "word_piip.png" 10;
"odsoo^" "symbols_odsoo^.png" 30;
"lcrzjb" "consonants_lcrzjb.png" 20;
"wcjrjq" "consonants_wcjrjq.png" 20;
"ähky" "word_ähky.png" 10;
"^vvdv" "symbols_^vvdv.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_^vvdv_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äänes" "word_äänes.png" 10;
"fotoni" "word_fotoni.png" 10;
"vtkdtx" "consonants_vtkdtx.png" 20;
"säie" "word_säie.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_säie_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uoma" "word_uoma.png" 10;
"uima" "word_uima.png" 10;
"älytä" "word_älytä.png" 10;
"sikhi" "word_sikhi.png" 10;
"vodv^^" "symbols_vodv^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _ _" "symbols_vodv^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"loraus" "word_loraus.png" 10;
"dod^os" "symbols_dod^os.png" 30;
"nieriä" "word_nieriä.png" 10;
"svvd" "symbols_svvd.png" 30;
"bgrzd" "consonants_bgrzd.png" 20;
"sqdq" "consonants_sqdq.png" 20;
"ovo^s" "symbols_ovo^s.png" 30;
"tyköä" "word_tyköä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_tyköä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mths" "consonants_mths.png" 20;
"vvv^" "symbols_vvv^.png" 30;
"mäntä" "word_mäntä.png" 10;
"ilmi" "word_ilmi.png" 10;
"vwrptk" "consonants_vwrptk.png" 20;
"noppa" "word_noppa.png" 10;
"uuni" "word_uuni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"n _ _ _" "word_uuni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"karies" "word_karies.png" 10;
"nurin" "word_nurin.png" 10;
"silsa" "word_silsa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ p _ _ _" "word_silsa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vsds" "symbols_vsds.png" 30;
"pidot" "word_pidot.png" 10;
"erkani" "word_erkani.png" 10;
"motata" "word_motata.png" 10;
"dwmdd" "consonants_dwmdd.png" 20;
"sarake" "word_sarake.png" 10;
"opaali" "word_opaali.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"o _ _ _ _ _" "word_opaali_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ratamo" "word_ratamo.png" 10;
"iäti" "word_iäti.png" 10;
"fwgntw" "consonants_fwgntw.png" 20;
"xgtt" "consonants_xgtt.png" 20;
"ilkiö" "word_ilkiö.png" 10;
"känsä" "word_känsä.png" 10;
"^dss" "symbols_^dss.png" 30;
"kortti" "word_kortti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t _ _" "word_kortti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdsdoo" "symbols_sdsdoo.png" 30;
"ssdvo^" "symbols_ssdvo^.png" 30;
"vahaus" "word_vahaus.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ s" "word_vahaus_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^ddv" "symbols_^ddv.png" 30;
"lohi" "word_lohi.png" 10;
"jymy" "word_jymy.png" 10;
"pöty" "word_pöty.png" 10;
"ahjo" "word_ahjo.png" 10;
"druidi" "word_druidi.png" 10;
"kiuas" "word_kiuas.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s" "word_kiuas_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rkgpjs" "consonants_rkgpjs.png" 20;
"osuma" "word_osuma.png" 10;
"menijä" "word_menijä.png" 10;
"zsgj" "consonants_zsgj.png" 20;
"gkzpg" "consonants_gkzpg.png" 20;
"sovssd" "symbols_sovssd.png" 30;
"sdvd^" "symbols_sdvd^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _" "symbols_sdvd^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gxqtx" "consonants_gxqtx.png" 20;
"gnxd" "consonants_gnxd.png" 20;
"ajos" "word_ajos.png" 10;
"sdosd" "symbols_sdosd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ s" "symbols_sdosd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kustos" "word_kustos.png" 10;
"vvovov" "symbols_vvovov.png" 30;
"häpy" "word_häpy.png" 10;
"ltqrr" "consonants_ltqrr.png" 20;
"myyty" "word_myyty.png" 10;
"ampuja" "word_ampuja.png" 10;
"näppy" "word_näppy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _" "word_näppy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"miten" "word_miten.png" 10;
"isyys" "word_isyys.png" 10;
"vodds" "symbols_vodds.png" 30;
"jaardi" "word_jaardi.png" 10;
"czkdbs" "consonants_czkdbs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _ _ _" "consonants_czkdbs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvvd" "symbols_vvvd.png" 30;
"pitäen" "word_pitäen.png" 10;
"jxfm" "consonants_jxfm.png" 20;
"riimu" "word_riimu.png" 10;
"xxqhnz" "consonants_xxqhnz.png" 20;
"v^vsov" "symbols_v^vsov.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _ _" "symbols_v^vsov_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"diodi" "word_diodi.png" 10;
"kopio" "word_kopio.png" 10;
"salaus" "word_salaus.png" 10;
"lhqt" "consonants_lhqt.png" 20;
"rwfcq" "consonants_rwfcq.png" 20;
"jäte" "word_jäte.png" 10;
"vlptrn" "consonants_vlptrn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ c _ _" "consonants_vlptrn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xggtw" "consonants_xggtw.png" 20;
"arpoa" "word_arpoa.png" 10;
"grgdd" "consonants_grgdd.png" 20;
"rfrqfp" "consonants_rfrqfp.png" 20;
"pesin" "word_pesin.png" 10;
"rtsgdh" "consonants_rtsgdh.png" 20;
"^s^^" "symbols_^s^^.png" 30;
"zkbwm" "consonants_zkbwm.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "consonants_zkbwm_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tcftcg" "consonants_tcftcg.png" 20;
"wvxw" "consonants_wvxw.png" 20;
"lldk" "consonants_lldk.png" 20;
"ttjglk" "consonants_ttjglk.png" 20;
"jänne" "word_jänne.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ p _" "word_jänne_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jakaja" "word_jakaja.png" 10;
"gcfj" "consonants_gcfj.png" 20;
"zkxtf" "consonants_zkxtf.png" 20;
"dovv^" "symbols_dovv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v _" "symbols_dovv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"seimi" "word_seimi.png" 10;
"sovs" "symbols_sovs.png" 30;
"lähi" "word_lähi.png" 10;
"oo^v" "symbols_oo^v.png" 30;
"voov^v" "symbols_voov^v.png" 30;
"ääriin" "word_ääriin.png" 10;
"häät" "word_häät.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ l" "word_häät_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vaje" "word_vaje.png" 10;
"jaos" "word_jaos.png" 10;
"terska" "word_terska.png" 10;
"jtltfj" "consonants_jtltfj.png" 20;
"itää" "word_itää.png" 10;
"uute" "word_uute.png" 10;
"^s^d" "symbols_^s^d.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _" "symbols_^s^d_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häkä" "word_häkä.png" 10;
"dsvvd" "symbols_dsvvd.png" 30;
"kopina" "word_kopina.png" 10;
"lgtjb" "consonants_lgtjb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ z" "consonants_lgtjb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"someen" "word_someen.png" 10;
"orpo" "word_orpo.png" 10;
"tämä" "word_tämä.png" 10;
"kaste" "word_kaste.png" 10;
"emätin" "word_emätin.png" 10;
"rukki" "word_rukki.png" 10;
"faksi" "word_faksi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _ _" "word_faksi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"d^v^^" "symbols_d^v^^.png" 30;
"pokeri" "word_pokeri.png" 10;
"rämä" "word_rämä.png" 10;
"möly" "word_möly.png" 10;
"^vdovd" "symbols_^vdovd.png" 30;
"dsdoo^" "symbols_dsdoo^.png" 30;
"mxnhh" "consonants_mxnhh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ x _ _ _" "consonants_mxnhh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyvi" "word_tyvi.png" 10;
"osinko" "word_osinko.png" 10;
"hqzll" "consonants_hqzll.png" 20;
"äänne" "word_äänne.png" 10;
"solmio" "word_solmio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_solmio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svks" "consonants_svks.png" 20;
"puida" "word_puida.png" 10;
"gsdx" "consonants_gsdx.png" 20;
"os^ood" "symbols_os^ood.png" 30;
"urut" "word_urut.png" 10;
"särkyä" "word_särkyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"j _ _ _ _ _" "word_särkyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ddvsd" "symbols_ddvsd.png" 30;
"zfjxqk" "consonants_zfjxqk.png" 20;
"shiia" "word_shiia.png" 10;
"ripeä" "word_ripeä.png" 10;
"vsv^" "symbols_vsv^.png" 30;
"kussa" "word_kussa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ü _" "word_kussa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"laukka" "word_laukka.png" 10;
"lempo" "word_lempo.png" 10;
"fuksi" "word_fuksi.png" 10;
"qmncn" "consonants_qmncn.png" 20;
"^dso" "symbols_^dso.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _" "symbols_^dso_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvv^v" "symbols_vvv^v.png" 30;
"sdsvs" "symbols_sdsvs.png" 30;
"pyörö" "word_pyörö.png" 10;
"psalmi" "word_psalmi.png" 10;
"dv^o" "symbols_dv^o.png" 30;
"harjus" "word_harjus.png" 10;
"säkä" "word_säkä.png" 10;
"xnbrh" "consonants_xnbrh.png" 20;
"ddjs" "consonants_ddjs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ j _" "consonants_ddjs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"d^^do" "symbols_d^^do.png" 30;
"lotja" "word_lotja.png" 10;
"s^o^^v" "symbols_s^o^^v.png" 30;
"bzhkd" "consonants_bzhkd.png" 20;
"odote" "word_odote.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ j" "word_odote_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nirso" "word_nirso.png" 10;
"itiö" "word_itiö.png" 10;
"kytkin" "word_kytkin.png" 10;
"suti" "word_suti.png" 10;
"xmpt" "consonants_xmpt.png" 20;
"wqghh" "consonants_wqghh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ h" "consonants_wqghh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^v^s" "symbols_s^v^s.png" 30;
"kysta" "word_kysta.png" 10;
"vsd^o" "symbols_vsd^o.png" 30;
"silaus" "word_silaus.png" 10;
"yöpyä" "word_yöpyä.png" 10;
"akti" "word_akti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _" "word_akti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äimä" "word_äimä.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"räme" "word_räme.png" 10;
"vuoka" "word_vuoka.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "word_vuoka_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sclc" "consonants_sclc.png" 20;
"köli" "word_köli.png" 10;
"hxjlgq" "consonants_hxjlgq.png" 20;
"oo^vdv" "symbols_oo^vdv.png" 30;
"^vvsvv" "symbols_^vvsvv.png" 30;
"do^s" "symbols_do^s.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "symbols_do^s_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hihna" "word_hihna.png" 10;
"qckq" "consonants_qckq.png" 20;
"ldtg" "consonants_ldtg.png" 20;
"kolhia" "word_kolhia.png" 10;
"näkö" "word_näkö.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
"kaapia" "word_kaapia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"k _ _ _ _ _" "word_kaapia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kohu" "word_kohu.png" 10;
"ässä" "word_ässä.png" 10;
"ds^^o" "symbols_ds^^o.png" 30;
"^s^sso" "symbols_^s^sso.png" 30;
"hioa" "word_hioa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _" "word_hioa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"duuri" "word_duuri.png" 10;
"^d^oos" "symbols_^d^oos.png" 30;
"kerubi" "word_kerubi.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
"zqshvj" "consonants_zqshvj.png" 20;
"jdfjs" "consonants_jdfjs.png" 20;
"tenä" "word_tenä.png" 10;
"sovdvo" "symbols_sovdvo.png" 30;
"dzvxq" "consonants_dzvxq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q" "consonants_dzvxq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gchqcz" "consonants_gchqcz.png" 20;
"rjfcdx" "consonants_rjfcdx.png" 20;
"brsjjh" "consonants_brsjjh.png" 20;
"bfkx" "consonants_bfkx.png" 20;
"nide" "word_nide.png" 10;
"kihu" "word_kihu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _" "word_kihu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^^^^" "symbols_v^^^^.png" 30;
"dodv" "symbols_dodv.png" 30;
"rulla" "word_rulla.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ w _ _" "word_rulla_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gqpkfs" "consonants_gqpkfs.png" 20;
"tykö" "word_tykö.png" 10;
"nrxlzx" "consonants_nrxlzx.png" 20;
"klaava" "word_klaava.png" 10;
"vssov" "symbols_vssov.png" 30;
"viipyä" "word_viipyä.png" 10;
"qpvbs" "consonants_qpvbs.png" 20;
"scgxmk" "consonants_scgxmk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ b _ _" "consonants_scgxmk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qwjvhz" "consonants_qwjvhz.png" 20;
"huorin" "word_huorin.png" 10;
"qvkjt" "consonants_qvkjt.png" 20;
"säle" "word_säle.png" 10;
"hamaan" "word_hamaan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ä _" "word_hamaan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jqdsmj" "consonants_jqdsmj.png" 20;
"sovo" "symbols_sovo.png" 30;
"poru" "word_poru.png" 10;
"lakea" "word_lakea.png" 10;
"säiky" "word_säiky.png" 10;
"kyhmy" "word_kyhmy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ h _" "word_kyhmy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gongi" "word_gongi.png" 10;
"tdgw" "consonants_tdgw.png" 20;
"vipu" "word_vipu.png" 10;
"tcfwdr" "consonants_tcfwdr.png" 20;
"o^v^" "symbols_o^v^.png" 30;
"torium" "word_torium.png" 10;
"sävy" "word_sävy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ e _" "word_sävy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xkfvmb" "consonants_xkfvmb.png" 20;
"täällä" "word_täällä.png" 10;
"svsod" "symbols_svsod.png" 30;
"sodvds" "symbols_sodvds.png" 30;
"ositus" "word_ositus.png" 10;
"nuha" "word_nuha.png" 10;
"hajan" "word_hajan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_hajan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kuje" "word_kuje.png" 10;
"lblxm" "consonants_lblxm.png" 20;
"o^ss" "symbols_o^ss.png" 30;
"tkcqd" "consonants_tkcqd.png" 20;
"kolvi" "word_kolvi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ü _ _ _" "word_kolvi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sv^vvs" "symbols_sv^vvs.png" 30;
"soov" "symbols_soov.png" 30;
"gmrq" "consonants_gmrq.png" 20;
"osdoo^" "symbols_osdoo^.png" 30;
"vs^oo" "symbols_vs^oo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_vs^oo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uuhi" "word_uuhi.png" 10;
"afasia" "word_afasia.png" 10;
"stoola" "word_stoola.png" 10;
"nczkbj" "consonants_nczkbj.png" 20;
"dlgt" "consonants_dlgt.png" 20;
"nurja" "word_nurja.png" 10;
"tyrä" "word_tyrä.png" 10;
"^o^so" "symbols_^o^so.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^o^so_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vovso" "symbols_vovso.png" 30;
"ndsvw" "consonants_ndsvw.png" 20;
"^s^vds" "symbols_^s^vds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _ _ _" "symbols_^s^vds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvsod" "symbols_vvsod.png" 30;
"räntä" "word_räntä.png" 10;
"vvo^" "symbols_vvo^.png" 30;
"kyteä" "word_kyteä.png" 10;
"^vd^vd" "symbols_^vd^vd.png" 30;
"s^dv^" "symbols_s^dv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _ _" "symbols_s^dv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vdod^" "symbols_vdod^.png" 30;
"txhs" "consonants_txhs.png" 20;
"katodi" "word_katodi.png" 10;
"vd^s" "symbols_vd^s.png" 30;
"blkxwz" "consonants_blkxwz.png" 20;
"^s^s" "symbols_^s^s.png" 30;
"otsoni" "word_otsoni.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
"ovvds" "symbols_ovvds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_ovvds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sopu" "word_sopu.png" 10;
"sodo" "symbols_sodo.png" 30;
"suippo" "word_suippo.png" 10;
"päkiä" "word_päkiä.png" 10;
"hautua" "word_hautua.png" 10;
"uuma" "word_uuma.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ u _ _" "word_uuma_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dfdzj" "consonants_dfdzj.png" 20;
"almu" "word_almu.png" 10;
"kuskus" "word_kuskus.png" 10;
"^o^o^v" "symbols_^o^o^v.png" 30;
"salvaa" "word_salvaa.png" 10;
"qmff" "consonants_qmff.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _" "consonants_qmff_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zlln" "consonants_zlln.png" 20;
"ylkä" "word_ylkä.png" 10;
"s^od" "symbols_s^od.png" 30;
"korren" "word_korren.png" 10;
"vvosod" "symbols_vvosod.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_vvosod_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"spmb" "consonants_spmb.png" 20;
"sddo" "symbols_sddo.png" 30;
"gnbtn" "consonants_gnbtn.png" 20;
"vv^^" "symbols_vv^^.png" 30;
"^osd" "symbols_^osd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ d _" "symbols_^osd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mpvl" "consonants_mpvl.png" 20;
"sdsosv" "symbols_sdsosv.png" 30;
};
