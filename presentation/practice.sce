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
"pöty" "word_pöty.png" 10;
"voo^o" "symbols_voo^o.png" 30;
"häpy" "word_häpy.png" 10;
"^vdovd" "symbols_^vdovd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ v _ _ _ _" "symbols_^vdovd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tämä" "word_tämä.png" 10;
"täällä" "word_täällä.png" 10;
"czkdbs" "consonants_czkdbs.png" 20;
"cnwh" "consonants_cnwh.png" 20;
"ssdvo^" "symbols_ssdvo^.png" 30;
"psalmi" "word_psalmi.png" 10;
"s^o^^v" "symbols_s^o^^v.png" 30;
"dovv^" "symbols_dovv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v _" "symbols_dovv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"qpvbs" "consonants_qpvbs.png" 20;
"sopu" "word_sopu.png" 10;
"^o^so" "symbols_^o^so.png" 30;
"emätin" "word_emätin.png" 10;
"kopina" "word_kopina.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ f _ _" "word_kopina_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"tcfwdr" "consonants_tcfwdr.png" 20;
"vodv^^" "symbols_vodv^^.png" 30;
"voov^v" "symbols_voov^v.png" 30;
"harjus" "word_harjus.png" 10;
"itiö" "word_itiö.png" 10;
"puida" "word_puida.png" 10;
"lotja" "word_lotja.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _ _" "word_lotja_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sovs" "symbols_sovs.png" 30;
"pidot" "word_pidot.png" 10;
"nczkbj" "consonants_nczkbj.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _ _ _" "consonants_nczkbj_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jtltfj" "consonants_jtltfj.png" 20;
"tenä" "word_tenä.png" 10;
"sovssd" "symbols_sovssd.png" 30;
"wqghh" "consonants_wqghh.png" 20;
"kyteä" "word_kyteä.png" 10;
"kolvi" "word_kolvi.png" 10;
"fuksi" "word_fuksi.png" 10;
"tdgw" "consonants_tdgw.png" 20;
"dsdoo^" "symbols_dsdoo^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ d" "symbols_dsdoo^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jakaja" "word_jakaja.png" 10;
"druidi" "word_druidi.png" 10;
"kaapia" "word_kaapia.png" 10;
"qmncn" "consonants_qmncn.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ m _ _ _" "consonants_qmncn_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^od" "symbols_s^od.png" 30;
"vvo^" "symbols_vvo^.png" 30;
"sclc" "consonants_sclc.png" 20;
"yöpyä" "word_yöpyä.png" 10;
"säkä" "word_säkä.png" 10;
"zfjxqk" "consonants_zfjxqk.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"z _ _ _ _ _" "consonants_zfjxqk_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ovo^s" "symbols_ovo^s.png" 30;
"sdsdoo" "symbols_sdsdoo.png" 30;
"salaus" "word_salaus.png" 10;
"kopio" "word_kopio.png" 10;
"sdosd" "symbols_sdosd.png" 30;
"brsjjh" "consonants_brsjjh.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ l" "consonants_brsjjh_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"säle" "word_säle.png" 10;
"kyhmy" "word_kyhmy.png" 10;
"qckq" "consonants_qckq.png" 20;
"^s^sso" "symbols_^s^sso.png" 30;
"ddvsd" "symbols_ddvsd.png" 30;
"^vd^vd" "symbols_^vd^vd.png" 30;
"rmmrh" "consonants_rmmrh.png" 20;
"dwmdd" "consonants_dwmdd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ d" "consonants_dwmdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvvd" "symbols_vvvd.png" 30;
"sovo" "symbols_sovo.png" 30;
"mpvl" "consonants_mpvl.png" 20;
"tyvi" "word_tyvi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ d _ _" "word_tyvi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"soov" "symbols_soov.png" 30;
"isyys" "word_isyys.png" 10;
"vahaus" "word_vahaus.png" 10;
"sv^vvs" "symbols_sv^vvs.png" 30;
"lähi" "word_lähi.png" 10;
"^svs^" "symbols_^svs^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_^svs^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"someen" "word_someen.png" 10;
"ksrwvk" "consonants_ksrwvk.png" 20;
"jxfm" "consonants_jxfm.png" 20;
"o^ss" "symbols_o^ss.png" 30;
"osdod^" "symbols_osdod^.png" 30;
"akti" "word_akti.png" 10;
"riimu" "word_riimu.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_riimu_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vovso" "symbols_vovso.png" 30;
"almu" "word_almu.png" 10;
"afasia" "word_afasia.png" 10;
"ampuja" "word_ampuja.png" 10;
"lgtjb" "consonants_lgtjb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ z" "consonants_lgtjb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"torium" "word_torium.png" 10;
"svds" "symbols_svds.png" 30;
"fwgntw" "consonants_fwgntw.png" 20;
"rukki" "word_rukki.png" 10;
"dod^os" "symbols_dod^os.png" 30;
"lemu" "word_lemu.png" 10;
"hormi" "word_hormi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ r _ _" "word_hormi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"känsä" "word_känsä.png" 10;
"jdfjs" "consonants_jdfjs.png" 20;
"vsd^o" "symbols_vsd^o.png" 30;
"pitäen" "word_pitäen.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_pitäen_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^v^" "symbols_v^v^.png" 30;
"sodo" "symbols_sodo.png" 30;
"ldtg" "consonants_ldtg.png" 20;
"ähky" "word_ähky.png" 10;
"svks" "consonants_svks.png" 20;
"os^ood" "symbols_os^ood.png" 30;
"gqpkfs" "consonants_gqpkfs.png" 20;
"sdsosv" "symbols_sdsosv.png" 30;
"klaava" "word_klaava.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ e _ _ _ _" "word_klaava_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vodds" "symbols_vodds.png" 30;
"tuohus" "word_tuohus.png" 10;
"häkä" "word_häkä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"h _ _ _" "word_häkä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kihu" "word_kihu.png" 10;
"sdvd^" "symbols_sdvd^.png" 30;
"odote" "word_odote.png" 10;
"ripeä" "word_ripeä.png" 10;
"xkfvmb" "consonants_xkfvmb.png" 20;
"gcfj" "consonants_gcfj.png" 20;
"hajan" "word_hajan.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_hajan_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"nide" "word_nide.png" 10;
"pesula" "word_pesula.png" 10;
"ääriin" "word_ääriin.png" 10;
"zkxtf" "consonants_zkxtf.png" 20;
"rjfcdx" "consonants_rjfcdx.png" 20;
"uuhi" "word_uuhi.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ z" "word_uuhi_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"syaani" "word_syaani.png" 10;
"duuri" "word_duuri.png" 10;
"köli" "word_köli.png" 10;
"motata" "word_motata.png" 10;
"ositus" "word_ositus.png" 10;
"ilmi" "word_ilmi.png" 10;
"erkani" "word_erkani.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ a _ _" "word_erkani_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"s^v^s" "symbols_s^v^s.png" 30;
"hjqc" "consonants_hjqc.png" 20;
"silaus" "word_silaus.png" 10;
"ajos" "word_ajos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"a _ _ _" "word_ajos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"svvd" "symbols_svvd.png" 30;
"menijä" "word_menijä.png" 10;
"kytkin" "word_kytkin.png" 10;
"dvsvos" "symbols_dvsvos.png" 30;
"siitos" "word_siitos.png" 10;
"xmpt" "consonants_xmpt.png" 20;
"hxjlgq" "consonants_hxjlgq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ _ x" "consonants_hxjlgq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^osd" "symbols_^osd.png" 30;
"vv^^" "symbols_vv^^.png" 30;
"s^dv^" "symbols_s^dv^.png" 30;
"qmff" "consonants_qmff.png" 20;
"vipu" "word_vipu.png" 10;
"osuma" "word_osuma.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ q _ _ _" "word_osuma_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gnbtn" "consonants_gnbtn.png" 20;
"zkwnr" "consonants_zkwnr.png" 20;
"hqzll" "consonants_hqzll.png" 20;
"kuskus" "word_kuskus.png" 10;
"myyty" "word_myyty.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"m _ _ _ _" "word_myyty_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"dzvxq" "consonants_dzvxq.png" 20;
"gsdx" "consonants_gsdx.png" 20;
"^vvdv" "symbols_^vvdv.png" 30;
"fotoni" "word_fotoni.png" 10;
"gkzpg" "consonants_gkzpg.png" 20;
"d^v^^" "symbols_d^v^^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"d _ _ _ _" "symbols_d^v^^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ilkiö" "word_ilkiö.png" 10;
"v^vsov" "symbols_v^vsov.png" 30;
"lempo" "word_lempo.png" 10;
"lohi" "word_lohi.png" 10;
"uute" "word_uute.png" 10;
"ylkä" "word_ylkä.png" 10;
"vssov" "symbols_vssov.png" 30;
"uuma" "word_uuma.png" 10;
"kaste" "word_kaste.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _" "word_kaste_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vovvvv" "symbols_vovvvv.png" 30;
"qwjvhz" "consonants_qwjvhz.png" 20;
"nurja" "word_nurja.png" 10;
"uivelo" "word_uivelo.png" 10;
"shiia" "word_shiia.png" 10;
"gmrq" "consonants_gmrq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "consonants_gmrq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"räntä" "word_räntä.png" 10;
"v^^^^" "symbols_v^^^^.png" 30;
"iäti" "word_iäti.png" 10;
"kuje" "word_kuje.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ z _" "word_kuje_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"gxqtx" "consonants_gxqtx.png" 20;
"sddo" "symbols_sddo.png" 30;
"luusto" "word_luusto.png" 10;
"sysi" "word_sysi.png" 10;
"dsvvd" "symbols_dsvvd.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ d" "symbols_dsvvd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xxqhnz" "consonants_xxqhnz.png" 20;
"älytä" "word_älytä.png" 10;
"estyä" "word_estyä.png" 10;
"vaje" "word_vaje.png" 10;
"vvv^" "symbols_vvv^.png" 30;
"sovdvo" "symbols_sovdvo.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ v _ _" "symbols_sovdvo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jäte" "word_jäte.png" 10;
"seimi" "word_seimi.png" 10;
"lblxm" "consonants_lblxm.png" 20;
"karies" "word_karies.png" 10;
"vs^oo" "symbols_vs^oo.png" 30;
"sarake" "word_sarake.png" 10;
"lldk" "consonants_lldk.png" 20;
"kustos" "word_kustos.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ ü _ _ _" "word_kustos_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"stoola" "word_stoola.png" 10;
"xgtt" "consonants_xgtt.png" 20;
"kerubi" "word_kerubi.png" 10;
"grgdd" "consonants_grgdd.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ x _ _ _" "consonants_grgdd_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"v^s^vd" "symbols_v^s^vd.png" 30;
"vlptrn" "consonants_vlptrn.png" 20;
"^d^oos" "symbols_^d^oos.png" 30;
"ndsvw" "consonants_ndsvw.png" 20;
"vtkdtx" "consonants_vtkdtx.png" 20;
"sävy" "word_sävy.png" 10;
"^vod" "symbols_^vod.png" 30;
"vd^s" "symbols_vd^s.png" 30;
"viipyä" "word_viipyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _ _" "word_viipyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"häät" "word_häät.png" 10;
"ssods" "symbols_ssods.png" 30;
"jqdsmj" "consonants_jqdsmj.png" 20;
"spmb" "consonants_spmb.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"q _ _ _" "consonants_spmb_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"näkö" "word_näkö.png" 10;
"bfkx" "consonants_bfkx.png" 20;
"hautua" "word_hautua.png" 10;
"kiuas" "word_kiuas.png" 10;
"mxnhh" "consonants_mxnhh.png" 20;
"rwfcq" "consonants_rwfcq.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ c _" "consonants_rwfcq_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"sdsvs" "symbols_sdsvs.png" 30;
"rfrqfp" "consonants_rfrqfp.png" 20;
"ltqrr" "consonants_ltqrr.png" 20;
"rkgpjs" "consonants_rkgpjs.png" 20;
"blkxwz" "consonants_blkxwz.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"b _ _ _ _ _" "consonants_blkxwz_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^ooso" "symbols_^ooso.png" 30;
"sikhi" "word_sikhi.png" 10;
"dlgt" "consonants_dlgt.png" 20;
"uoma" "word_uoma.png" 10;
"mäntä" "word_mäntä.png" 10;
"^vvsvv" "symbols_^vvsvv.png" 30;
"d^^do" "symbols_d^^do.png" 30;
"eliö" "word_eliö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _" "word_eliö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"faksi" "word_faksi.png" 10;
"wvxw" "consonants_wvxw.png" 20;
"osinko" "word_osinko.png" 10;
"juhta" "word_juhta.png" 10;
"kolhia" "word_kolhia.png" 10;
"vsv^" "symbols_vsv^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ s _ _" "symbols_vsv^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"oo^v" "symbols_oo^v.png" 30;
"xnbrh" "consonants_xnbrh.png" 20;
"hioa" "word_hioa.png" 10;
"jänne" "word_jänne.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ p _" "word_jänne_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kussa" "word_kussa.png" 10;
"zpcqc" "consonants_zpcqc.png" 20;
"hnxf" "consonants_hnxf.png" 20;
"o^v^" "symbols_o^v^.png" 30;
"oo^vdv" "symbols_oo^vdv.png" 30;
"^s^vds" "symbols_^s^vds.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"^ _ _ _ _ _" "symbols_^s^vds_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"jymy" "word_jymy.png" 10;
"ässä" "word_ässä.png" 10;
"tykö" "word_tykö.png" 10;
"^o^o^v" "symbols_^o^o^v.png" 30;
"uuni" "word_uuni.png" 10;
"äänne" "word_äänne.png" 10;
"gnxd" "consonants_gnxd.png" 20;
"vwrptk" "consonants_vwrptk.png" 20;
"tcftcg" "consonants_tcftcg.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ n _ _" "consonants_tcftcg_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"kohu" "word_kohu.png" 10;
"diodi" "word_diodi.png" 10;
"katodi" "word_katodi.png" 10;
"zkbwm" "consonants_zkbwm.png" 20;
"^s^^" "symbols_^s^^.png" 30;
"orpo" "word_orpo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ ö _ _" "word_orpo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rämä" "word_rämä.png" 10;
"kysta" "word_kysta.png" 10;
"sqdq" "consonants_sqdq.png" 20;
"pokeri" "word_pokeri.png" 10;
"huorin" "word_huorin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"g _ _ _ _ _" "word_huorin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äänes" "word_äänes.png" 10;
"vvosod" "symbols_vvosod.png" 30;
"ovvds" "symbols_ovvds.png" 30;
"näppy" "word_näppy.png" 10;
"vuoka" "word_vuoka.png" 10;
"salvaa" "word_salvaa.png" 10;
"säiky" "word_säiky.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ i _ _" "word_säiky_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ds^^o" "symbols_ds^^o.png" 30;
"noppa" "word_noppa.png" 10;
"suti" "word_suti.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ z _ _" "word_suti_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"uiva" "word_uiva.png" 10;
"jaardi" "word_jaardi.png" 10;
"jaos" "word_jaos.png" 10;
"gchqcz" "consonants_gchqcz.png" 20;
"ddjs" "consonants_ddjs.png" 20;
"möly" "word_möly.png" 10;
"kortti" "word_kortti.png" 10;
"oinas" "word_oinas.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"o _ _ _ _" "word_oinas_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"äimä" "word_äimä.png" 10;
"^s^d" "symbols_^s^d.png" 30;
"tyrä" "word_tyrä.png" 10;
"zjwhs" "consonants_zjwhs.png" 20;
"tkcqd" "consonants_tkcqd.png" 20;
"odsoo^" "symbols_odsoo^.png" 30;
"silsa" "word_silsa.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ p _ _ _" "word_silsa_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"loraus" "word_loraus.png" 10;
"^d^vs" "symbols_^d^vs.png" 30;
"nuha" "word_nuha.png" 10;
"rtsgdh" "consonants_rtsgdh.png" 20;
"lakea" "word_lakea.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ e _" "word_lakea_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"otsoni" "word_otsoni.png" 10;
"^dss" "symbols_^dss.png" 30;
"itää" "word_itää.png" 10;
"^s^s" "symbols_^s^s.png" 30;
"pamppu" "word_pamppu.png" 10;
"miten" "word_miten.png" 10;
"ratamo" "word_ratamo.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _ _ _" "word_ratamo_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"särkyä" "word_särkyä.png" 10;
"suippo" "word_suippo.png" 10;
"qvkjt" "consonants_qvkjt.png" 20;
"nirso" "word_nirso.png" 10;
"laukka" "word_laukka.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ u _ _ _" "word_laukka_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"rulla" "word_rulla.png" 10;
"sodvds" "symbols_sodvds.png" 30;
"korren" "word_korren.png" 10;
"lhqt" "consonants_lhqt.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ t" "consonants_lhqt_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"urut" "word_urut.png" 10;
"vvv^v" "symbols_vvv^v.png" 30;
"dv^o" "symbols_dv^o.png" 30;
"bzhkd" "consonants_bzhkd.png" 20;
"terska" "word_terska.png" 10;
"uima" "word_uima.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"u _ _ _" "word_uima_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"lcrzjb" "consonants_lcrzjb.png" 20;
"mths" "consonants_mths.png" 20;
"zdvv" "consonants_zdvv.png" 20;
"scgxmk" "consonants_scgxmk.png" 20;
"nieriä" "word_nieriä.png" 10;
"zlln" "consonants_zlln.png" 20;
};

TEMPLATE "question.tem" {
question file code;
"_ l _ _" "consonants_zlln_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"hihna" "word_hihna.png" 10;
"dodv" "symbols_dodv.png" 30;
"osdoo^" "symbols_osdoo^.png" 30;
"zsgj" "consonants_zsgj.png" 20;
"nrxlzx" "consonants_nrxlzx.png" 20;
"bgrzd" "consonants_bgrzd.png" 20;
"syöpyä" "word_syöpyä.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ a _ _ _ _" "word_syöpyä_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ripsi" "word_ripsi.png" 10;
"räme" "word_räme.png" 10;
"tyköä" "word_tyköä.png" 10;
"opaali" "word_opaali.png" 10;
"päkiä" "word_päkiä.png" 10;
"arpoa" "word_arpoa.png" 10;
"txhs" "consonants_txhs.png" 20;
"vdod^" "symbols_vdod^.png" 30;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ d _" "symbols_vdod^_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"^ddv" "symbols_^ddv.png" 30;
"rapsi" "word_rapsi.png" 10;
"poru" "word_poru.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ u" "word_poru_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"xggtw" "consonants_xggtw.png" 20;
"svsod" "symbols_svsod.png" 30;
"piip" "word_piip.png" 10;
"dfdzj" "consonants_dfdzj.png" 20;
"säie" "word_säie.png" 10;
"ttjglk" "consonants_ttjglk.png" 20;
"pesin" "word_pesin.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ _ _ n" "word_pesin_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"ahjo" "word_ahjo.png" 10;
"vvovov" "symbols_vvovov.png" 30;
"vsds" "symbols_vsds.png" 30;
"zqshvj" "consonants_zqshvj.png" 20;
"^dso" "symbols_^dso.png" 30;
"wcjrjq" "consonants_wcjrjq.png" 20;
"nisä" "word_nisä.png" 10;
"pyörö" "word_pyörö.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ t _ _" "word_pyörö_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"do^s" "symbols_do^s.png" 30;
"hamaan" "word_hamaan.png" 10;
"nurin" "word_nurin.png" 10;
"solmio" "word_solmio.png" 10;
};

TEMPLATE "question.tem" {
question file code;
"_ _ l _ _ _" "word_solmio_question.png" 40;
};

TEMPLATE "stimulus.tem" {
word file code;
"vvsod" "symbols_vvsod.png" 30;
"gongi" "word_gongi.png" 10;
};
