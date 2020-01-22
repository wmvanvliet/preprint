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
"tdgw" "consonants_tdgw.png" 20;
"uuma" "word_uuma.png" 10;
"sdsdoo" "symbols_sdsdoo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _ _ _" "symbols_sdsdoo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^s^d" "symbols_^s^d.png" 30;
"otsoni" "word_otsoni.png" 10;
"tenä" "word_tenä.png" 10;
"akti" "word_akti.png" 10;
"kuje" "word_kuje.png" 10;
"oo^v" "symbols_oo^v.png" 30;
"mpvl" "consonants_mpvl.png" 20;
"dvsvos" "symbols_dvsvos.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _ _" "symbols_dvsvos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rjfcdx" "consonants_rjfcdx.png" 20;
"^o^o^v" "symbols_^o^o^v.png" 30;
"hioa" "word_hioa.png" 10;
"syöpyä" "word_syöpyä.png" 10;
"kolhia" "word_kolhia.png" 10;
"^d^oos" "symbols_^d^oos.png" 30;
"bgrzd" "consonants_bgrzd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"b _ _ _ _" "consonants_bgrzd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vlptrn" "consonants_vlptrn.png" 20;
"älytä" "word_älytä.png" 10;
"uute" "word_uute.png" 10;
"säle" "word_säle.png" 10;
"hajan" "word_hajan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_hajan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kaste" "word_kaste.png" 10;
"mäntä" "word_mäntä.png" 10;
"ds^^o" "symbols_ds^^o.png" 30;
"hjqc" "consonants_hjqc.png" 20;
"loraus" "word_loraus.png" 10;
"erkani" "word_erkani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_erkani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"möly" "word_möly.png" 10;
"hamaan" "word_hamaan.png" 10;
"someen" "word_someen.png" 10;
"kihu" "word_kihu.png" 10;
"brsjjh" "consonants_brsjjh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ l" "consonants_brsjjh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"laukka" "word_laukka.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"nieriä" "word_nieriä.png" 10;
"huorin" "word_huorin.png" 10;
"dv^o" "symbols_dv^o.png" 30;
"sodvds" "symbols_sodvds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_sodvds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"o^v^" "symbols_o^v^.png" 30;
"vahaus" "word_vahaus.png" 10;
"kortti" "word_kortti.png" 10;
"nisä" "word_nisä.png" 10;
"stoola" "word_stoola.png" 10;
"dodv" "symbols_dodv.png" 30;
"vvv^v" "symbols_vvv^v.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "symbols_vvv^v_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rkgpjs" "consonants_rkgpjs.png" 20;
"kyteä" "word_kyteä.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"v^vsov" "symbols_v^vsov.png" 30;
"jänne" "word_jänne.png" 10;
"sqdq" "consonants_sqdq.png" 20;
"lgtjb" "consonants_lgtjb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ z" "consonants_lgtjb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lähi" "word_lähi.png" 10;
"rmmrh" "consonants_rmmrh.png" 20;
"äimä" "word_äimä.png" 10;
"uivelo" "word_uivelo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ o" "word_uivelo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vwrptk" "consonants_vwrptk.png" 20;
"bfkx" "consonants_bfkx.png" 20;
"ajos" "word_ajos.png" 10;
"vovso" "symbols_vovso.png" 30;
"voo^o" "symbols_voo^o.png" 30;
"ldtg" "consonants_ldtg.png" 20;
"ylkä" "word_ylkä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ k _" "word_ylkä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"do^s" "symbols_do^s.png" 30;
"häät" "word_häät.png" 10;
"zkwnr" "consonants_zkwnr.png" 20;
"ratamo" "word_ratamo.png" 10;
"tcfwdr" "consonants_tcfwdr.png" 20;
"almu" "word_almu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"a _ _ _" "word_almu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dlgt" "consonants_dlgt.png" 20;
"lcrzjb" "consonants_lcrzjb.png" 20;
"jdfjs" "consonants_jdfjs.png" 20;
"svsod" "symbols_svsod.png" 30;
"lhqt" "consonants_lhqt.png" 20;
"lldk" "consonants_lldk.png" 20;
"pamppu" "word_pamppu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ q _" "word_pamppu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pitäen" "word_pitäen.png" 10;
"dovv^" "symbols_dovv^.png" 30;
"tcftcg" "consonants_tcftcg.png" 20;
"jymy" "word_jymy.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ y" "word_jymy_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"eliö" "word_eliö.png" 10;
"blkxwz" "consonants_blkxwz.png" 20;
"pyörö" "word_pyörö.png" 10;
"^ooso" "symbols_^ooso.png" 30;
"kustos" "word_kustos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ü _ _ _" "word_kustos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"grgdd" "consonants_grgdd.png" 20;
"näppy" "word_näppy.png" 10;
"gsdx" "consonants_gsdx.png" 20;
"vd^s" "symbols_vd^s.png" 30;
"osinko" "word_osinko.png" 10;
"piip" "word_piip.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ i _ _" "word_piip_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"särkyä" "word_särkyä.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
"^s^s" "symbols_^s^s.png" 30;
"odsoo^" "symbols_odsoo^.png" 30;
"torium" "word_torium.png" 10;
"räntä" "word_räntä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"r _ _ _ _" "word_räntä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"mths" "consonants_mths.png" 20;
"zdvv" "consonants_zdvv.png" 20;
"d^v^^" "symbols_d^v^^.png" 30;
"oo^vdv" "symbols_oo^vdv.png" 30;
"^svs^" "symbols_^svs^.png" 30;
"estyä" "word_estyä.png" 10;
"jxfm" "consonants_jxfm.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ x" "consonants_jxfm_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"os^ood" "symbols_os^ood.png" 30;
"sysi" "word_sysi.png" 10;
"päkiä" "word_päkiä.png" 10;
"vovvvv" "symbols_vovvvv.png" 30;
"tämä" "word_tämä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ä _ _" "word_tämä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vodv^^" "symbols_vodv^^.png" 30;
"gxqtx" "consonants_gxqtx.png" 20;
"jäte" "word_jäte.png" 10;
"lemu" "word_lemu.png" 10;
"korren" "word_korren.png" 10;
"v^^^^" "symbols_v^^^^.png" 30;
"sovs" "symbols_sovs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _" "symbols_sovs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"räme" "word_räme.png" 10;
"cnwh" "consonants_cnwh.png" 20;
"ampuja" "word_ampuja.png" 10;
"kyhmy" "word_kyhmy.png" 10;
"ässä" "word_ässä.png" 10;
"gcfj" "consonants_gcfj.png" 20;
"s^o^^v" "symbols_s^o^^v.png" 30;
"hihna" "word_hihna.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ü _" "word_hihna_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"terska" "word_terska.png" 10;
"siitos" "word_siitos.png" 10;
"^o^so" "symbols_^o^so.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ o _ _" "symbols_^o^so_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zfjxqk" "consonants_zfjxqk.png" 20;
"säiky" "word_säiky.png" 10;
"luusto" "word_luusto.png" 10;
"sovdvo" "symbols_sovdvo.png" 30;
"sovssd" "symbols_sovssd.png" 30;
"poru" "word_poru.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ u" "word_poru_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"osuma" "word_osuma.png" 10;
"yöpyä" "word_yöpyä.png" 10;
"spmb" "consonants_spmb.png" 20;
"rwfcq" "consonants_rwfcq.png" 20;
"nurin" "word_nurin.png" 10;
"xggtw" "consonants_xggtw.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ w" "consonants_xggtw_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^vdovd" "symbols_^vdovd.png" 30;
"^vod" "symbols_^vod.png" 30;
"afasia" "word_afasia.png" 10;
"sikhi" "word_sikhi.png" 10;
"sopu" "word_sopu.png" 10;
"uima" "word_uima.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_uima_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lakea" "word_lakea.png" 10;
"psalmi" "word_psalmi.png" 10;
"seimi" "word_seimi.png" 10;
"soov" "symbols_soov.png" 30;
"ovo^s" "symbols_ovo^s.png" 30;
"opaali" "word_opaali.png" 10;
"orpo" "word_orpo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ö _ _" "word_orpo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sddo" "symbols_sddo.png" 30;
"sdsosv" "symbols_sdsosv.png" 30;
"ähky" "word_ähky.png" 10;
"ovvds" "symbols_ovvds.png" 30;
"vvosod" "symbols_vvosod.png" 30;
"^vd^vd" "symbols_^vd^vd.png" 30;
"gqpkfs" "consonants_gqpkfs.png" 20;
"lohi" "word_lohi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ q _" "word_lohi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kaapia" "word_kaapia.png" 10;
"ddvsd" "symbols_ddvsd.png" 30;
"kiuas" "word_kiuas.png" 10;
"dsdoo^" "symbols_dsdoo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_dsdoo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"juhta" "word_juhta.png" 10;
"czkdbs" "consonants_czkdbs.png" 20;
"sv^vvs" "symbols_sv^vvs.png" 30;
"äänne" "word_äänne.png" 10;
"gnxd" "consonants_gnxd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"b _ _ _" "consonants_gnxd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qmncn" "consonants_qmncn.png" 20;
"odote" "word_odote.png" 10;
"ilmi" "word_ilmi.png" 10;
"ahjo" "word_ahjo.png" 10;
"scgxmk" "consonants_scgxmk.png" 20;
"näkö" "word_näkö.png" 10;
"gnbtn" "consonants_gnbtn.png" 20;
"dsvvd" "symbols_dsvvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ d" "symbols_dsvvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"bzhkd" "consonants_bzhkd.png" 20;
"wvxw" "consonants_wvxw.png" 20;
"vsds" "symbols_vsds.png" 30;
"ilkiö" "word_ilkiö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ ö" "word_ilkiö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kytkin" "word_kytkin.png" 10;
"vaje" "word_vaje.png" 10;
"häkä" "word_häkä.png" 10;
"wcjrjq" "consonants_wcjrjq.png" 20;
"vipu" "word_vipu.png" 10;
"säkä" "word_säkä.png" 10;
"jaardi" "word_jaardi.png" 10;
"ndsvw" "consonants_ndsvw.png" 20;
"sdsvs" "symbols_sdsvs.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _ _" "symbols_sdsvs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvvd" "symbols_vvvd.png" 30;
"gkzpg" "consonants_gkzpg.png" 20;
"silaus" "word_silaus.png" 10;
"rämä" "word_rämä.png" 10;
"jaos" "word_jaos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ c _" "word_jaos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qpvbs" "consonants_qpvbs.png" 20;
"iäti" "word_iäti.png" 10;
"kysta" "word_kysta.png" 10;
"karies" "word_karies.png" 10;
"shiia" "word_shiia.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _" "word_shiia_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"klaava" "word_klaava.png" 10;
"salaus" "word_salaus.png" 10;
"xkfvmb" "consonants_xkfvmb.png" 20;
"ripsi" "word_ripsi.png" 10;
"osdod^" "symbols_osdod^.png" 30;
"tyrä" "word_tyrä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_tyrä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^osd" "symbols_^osd.png" 30;
"qvkjt" "consonants_qvkjt.png" 20;
"mxnhh" "consonants_mxnhh.png" 20;
"hnxf" "consonants_hnxf.png" 20;
"vtkdtx" "consonants_vtkdtx.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ c _ _ _" "consonants_vtkdtx_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^dso" "symbols_^dso.png" 30;
"svvd" "symbols_svvd.png" 30;
"wqghh" "consonants_wqghh.png" 20;
"voov^v" "symbols_voov^v.png" 30;
"svds" "symbols_svds.png" 30;
"o^ss" "symbols_o^ss.png" 30;
"nide" "word_nide.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ m" "word_nide_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ddjs" "consonants_ddjs.png" 20;
"äänes" "word_äänes.png" 10;
"uoma" "word_uoma.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
"s^od" "symbols_s^od.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ o" "symbols_s^od_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tkcqd" "consonants_tkcqd.png" 20;
"uuni" "word_uuni.png" 10;
"jakaja" "word_jakaja.png" 10;
"ääriin" "word_ääriin.png" 10;
"vv^^" "symbols_vv^^.png" 30;
"gongi" "word_gongi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _" "word_gongi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tyköä" "word_tyköä.png" 10;
"känsä" "word_känsä.png" 10;
"kerubi" "word_kerubi.png" 10;
"svks" "consonants_svks.png" 20;
"kolvi" "word_kolvi.png" 10;
"zkbwm" "consonants_zkbwm.png" 20;
"urut" "word_urut.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_urut_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"syaani" "word_syaani.png" 10;
"arpoa" "word_arpoa.png" 10;
"itiö" "word_itiö.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
"pokeri" "word_pokeri.png" 10;
"lotja" "word_lotja.png" 10;
"häpy" "word_häpy.png" 10;
"dfdzj" "consonants_dfdzj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_dfdzj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"viipyä" "word_viipyä.png" 10;
"vdod^" "symbols_vdod^.png" 30;
"riimu" "word_riimu.png" 10;
"uiva" "word_uiva.png" 10;
"rukki" "word_rukki.png" 10;
"faksi" "word_faksi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ p _ _" "word_faksi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xxqhnz" "consonants_xxqhnz.png" 20;
"vvovov" "symbols_vvovov.png" 30;
"hxjlgq" "consonants_hxjlgq.png" 20;
"vodds" "symbols_vodds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _" "symbols_vodds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"d^^do" "symbols_d^^do.png" 30;
"tyvi" "word_tyvi.png" 10;
"vvsod" "symbols_vvsod.png" 30;
"dzvxq" "consonants_dzvxq.png" 20;
"suippo" "word_suippo.png" 10;
"sävy" "word_sävy.png" 10;
"dod^os" "symbols_dod^os.png" 30;
"hormi" "word_hormi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _ _" "word_hormi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gmrq" "consonants_gmrq.png" 20;
"myyty" "word_myyty.png" 10;
"ositus" "word_ositus.png" 10;
"motata" "word_motata.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"x _ _ _ _ _" "word_motata_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"itää" "word_itää.png" 10;
"rulla" "word_rulla.png" 10;
"emätin" "word_emätin.png" 10;
"s^dv^" "symbols_s^dv^.png" 30;
"noppa" "word_noppa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ ü _" "word_noppa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sovo" "symbols_sovo.png" 30;
"pidot" "word_pidot.png" 10;
"lempo" "word_lempo.png" 10;
"fwgntw" "consonants_fwgntw.png" 20;
"jtltfj" "consonants_jtltfj.png" 20;
"puida" "word_puida.png" 10;
"druidi" "word_druidi.png" 10;
"ssdvo^" "symbols_ssdvo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"s _ _ _ _ _" "symbols_ssdvo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sodo" "symbols_sodo.png" 30;
"^ddv" "symbols_^ddv.png" 30;
"nuha" "word_nuha.png" 10;
"ssods" "symbols_ssods.png" 30;
"^vvdv" "symbols_^vvdv.png" 30;
"täällä" "word_täällä.png" 10;
"vuoka" "word_vuoka.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"v _ _ _ _" "word_vuoka_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nczkbj" "consonants_nczkbj.png" 20;
"kohu" "word_kohu.png" 10;
"^dss" "symbols_^dss.png" 30;
"^s^^" "symbols_^s^^.png" 30;
"vs^oo" "symbols_vs^oo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_vs^oo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"fuksi" "word_fuksi.png" 10;
"gchqcz" "consonants_gchqcz.png" 20;
"sarake" "word_sarake.png" 10;
"qckq" "consonants_qckq.png" 20;
"ltqrr" "consonants_ltqrr.png" 20;
"qmff" "consonants_qmff.png" 20;
"xnbrh" "consonants_xnbrh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"x _ _ _ _" "consonants_xnbrh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"miten" "word_miten.png" 10;
"s^v^s" "symbols_s^v^s.png" 30;
"zsgj" "consonants_zsgj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ g _" "consonants_zsgj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nirso" "word_nirso.png" 10;
"ripeä" "word_ripeä.png" 10;
"isyys" "word_isyys.png" 10;
"kuskus" "word_kuskus.png" 10;
"salvaa" "word_salvaa.png" 10;
"osdoo^" "symbols_osdoo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ o _" "symbols_osdoo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rfrqfp" "consonants_rfrqfp.png" 20;
"nrxlzx" "consonants_nrxlzx.png" 20;
"zkxtf" "consonants_zkxtf.png" 20;
"^s^sso" "symbols_^s^sso.png" 30;
"säie" "word_säie.png" 10;
"silsa" "word_silsa.png" 10;
"qwjvhz" "consonants_qwjvhz.png" 20;
"txhs" "consonants_txhs.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ h _" "consonants_txhs_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vssov" "symbols_vssov.png" 30;
"hqzll" "consonants_hqzll.png" 20;
"köli" "word_köli.png" 10;
"vsd^o" "symbols_vsd^o.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "symbols_vsd^o_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pöty" "word_pöty.png" 10;
"diodi" "word_diodi.png" 10;
"lblxm" "consonants_lblxm.png" 20;
"menijä" "word_menijä.png" 10;
"sdosd" "symbols_sdosd.png" 30;
"solmio" "word_solmio.png" 10;
"katodi" "word_katodi.png" 10;
"tuohus" "word_tuohus.png" 10;
"uuhi" "word_uuhi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ z" "word_uuhi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kopina" "word_kopina.png" 10;
"sclc" "consonants_sclc.png" 20;
"oinas" "word_oinas.png" 10;
"rapsi" "word_rapsi.png" 10;
"harjus" "word_harjus.png" 10;
"hautua" "word_hautua.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _ _" "word_hautua_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"pesin" "word_pesin.png" 10;
"^s^vds" "symbols_^s^vds.png" 30;
"^vvsvv" "symbols_^vvsvv.png" 30;
"kussa" "word_kussa.png" 10;
"suti" "word_suti.png" 10;
"dwmdd" "consonants_dwmdd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ d" "consonants_dwmdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"zjwhs" "consonants_zjwhs.png" 20;
"vsv^" "symbols_vsv^.png" 30;
"fotoni" "word_fotoni.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"e _ _ _ _ _" "word_fotoni_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdvd^" "symbols_sdvd^.png" 30;
"v^v^" "symbols_v^v^.png" 30;
"duuri" "word_duuri.png" 10;
"jqdsmj" "consonants_jqdsmj.png" 20;
"vvv^" "symbols_vvv^.png" 30;
"kopio" "word_kopio.png" 10;
"tykö" "word_tykö.png" 10;
"v^s^vd" "symbols_v^s^vd.png" 30;
"zlln" "consonants_zlln.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ l _ _" "consonants_zlln_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xmpt" "consonants_xmpt.png" 20;
"nurja" "word_nurja.png" 10;
"pesula" "word_pesula.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_pesula_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvo^" "symbols_vvo^.png" 30;
"rtsgdh" "consonants_rtsgdh.png" 20;
"zqshvj" "consonants_zqshvj.png" 20;
};
